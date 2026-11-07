"""Which extra inputs improve the lift-anchored forecast? Same protocol as lift_backtest.py
(forecast issued N days before every real lift), but each experiment is a model plus a set of
extra inputs:

    base model      timestral_cum | timesfm3_cum | timestral | timesfm3 | ramp | persistence
    @N              context length in days (default 30)
    +how            known future: hour-of-week sin/cos + weekend flag (local time)
    +cal            known future: site calendar, open-this-hour and open-this-day flags
    +cycles         past only: compaction cycles per hour
    +mass           past only: container mass with lifts undone (cumulative)
    +fill           past only: fill_pct with lifts undone (for --target mass_kg)

--target mass_kg forecasts the weight-sensor mass instead of fill_pct (devices without a scale
are skipped; lifts are detected as drops larger than 30 % of the series' 95th percentile) and
also scores the forecast against the weighbridge ticket of the lift.

Example:
    uv run python covariate_test.py --leads 2,4 \
        --experiments timestral_cum,timestral_cum+how,timestral_cum+cal,timestral_cum+cal+how,timestral_cum+cycles,timestral_cum+mass,timestral_cum@60,timestral_cum@60+cal
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import build_models
from fill_forecast import db
from fill_forecast.covariates import hour_of_week, site_open_hours, undo_resets
from fill_forecast.metrics import first_crossing_hours
from fill_forecast.models import QuantileForecast
from lift_backtest import RESET_STATUSES, utc

DEFAULT_EXPERIMENTS = ("timestral_cum,timestral_cum+how,timestral_cum+cal,timestral_cum+cal+how,"
                       "timestral_cum+cycles,timestral_cum+mass,timestral_cum+cal+cycles,timestral_cum@60,timestral_cum@60+cal,"
                       "timesfm3_cum,timesfm3_cum+cal+how,ramp")


def parse_experiment(spec: str) -> dict:
    m = re.match(r"^([a-z0-9_]+)(?:@(\d+))?((?:\+[a-z]+)*)$", spec)
    if not m:
        raise SystemExit(f"bad experiment spec {spec!r}")
    feats = [f for f in m.group(3).split("+") if f]
    return {"spec": spec, "model": m.group(1), "context_days": int(m.group(2) or 30),
            "known": [f for f in feats if f in ("how", "cal")],
            "past": [f for f in feats if f in ("cycles", "mass", "fill")]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices")
    p.add_argument("--from", dest="date_from", default="2026-06-01")
    p.add_argument("--to", dest="date_to", default="2026-10-01")
    p.add_argument("--leads", default="2,4")
    p.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    p.add_argument("--timestral-repo", default="mistralai/Timestral")
    p.add_argument("--timesfm-checkpoint", default="google/timesfm-3.0-pytorch")
    p.add_argument("--device-type", default="cpu")
    p.add_argument("--schema", default="public", choices=db.SCHEMAS)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--target", default="fill_pct", choices=("fill_pct", "mass_kg"))
    p.add_argument("--out", default="outputs/covariate_test")
    args = p.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")]
    exps = [parse_experiment(s.strip()) for s in args.experiments.split(",") if s.strip()]
    target = args.target
    args.reset_frac = 0.3 if target == "mass_kg" else None
    if target == "mass_kg":
        from fill_forecast.models import set_value_range
        set_value_range(0.0, float("inf"))
    max_ctx = max(e["context_days"] for e in exps)
    date_from, date_to = utc(args.date_from), utc(args.date_to)
    conn = db.connect()

    with conn.cursor() as cur:
        if args.devices:
            devices = [d.strip() for d in args.devices.split(",")]
        else:
            cur.execute(f"SELECT device_id FROM {args.schema}.devices ORDER BY 1")
            devices = [r[0] for r in cur.fetchall()]
    metas = {d: db.fetch_device_meta(conn, d, args.schema) for d in devices}
    if target == "mass_kg":
        devices = [d for d in devices if metas[d].get("has_weight_measurement", True) is not False]
    events = []
    for d in devices:
        ev = db.fetch_collection_events(conn, d, date_from, date_to, args.schema)
        ev = ev[ev["status"].isin(RESET_STATUSES)]
        for _, r in ev.iterrows():
            events.append({"device_id": d, "lift_at": r["actual_at"], "trigger_type": r["trigger_type"],
                           "event_mass_before": r["device_mass_kg_before"], "event_collected_kg": r["device_estimated_collected_kg"],
                           "weighbridge_net_kg": r["weighbridge_net_kg"]})
    events = pd.DataFrame(events).sort_values(["device_id", "lift_at"]).reset_index(drop=True)
    if args.max_events:
        events = events.head(args.max_events)
    events["prev_lift_at"] = events.groupby("device_id")["lift_at"].shift(1)
    print(f"{len(events)} lifts on {len(devices)} devices, leads {leads}, {len(exps)} experiments")

    t0 = time.perf_counter()
    frames = {d: db.fetch_hourly_frame(conn, d, date_from - timedelta(days=max_ctx + max(leads) + 1), date_to, args.schema)
              for d in devices}
    print(f"fetched {len(devices)} hourly frames in {time.perf_counter() - t0:.1f}s")

    # model instances are shared between experiments with the same base model
    model_cache: dict[str, object] = {}

    def get_model(name):
        if name not in model_cache:
            model_cache[name] = build_models([name], args)[name]
        return model_cache[name]

    rows = []
    for lead in leads:
        H = lead * 24
        for e in exps:
            ctx_n = e["context_days"] * 24
            ctx_l, act_l, known_l, past_l, keys = [], [], [], [], []
            for i, ev in events.iterrows():
                f = frames[ev.device_id]
                lift_hour = ev.lift_at.floor("h")
                cutoff = lift_hour - pd.Timedelta(hours=H)
                start = cutoff - pd.Timedelta(hours=ctx_n)
                c = f[(f.index >= start) & (f.index < cutoff)]
                a = f[(f.index >= cutoff) & (f.index < lift_hour)]
                if len(c) != ctx_n or len(a) != H or c[target].isna().mean() > 0.3 or a[target].notna().sum() == 0:
                    continue
                full_idx = c.index.append(a.index)
                known = []
                if "how" in e["known"]:
                    known.append(hour_of_week(full_idx))
                if "cal" in e["known"]:
                    known.append(site_open_hours(metas[ev.device_id].get("site_id"), full_idx))
                past = []
                if "cycles" in e["past"]:
                    past.append(c.compaction_cycles.fillna(0).to_numpy(np.float32)[None])
                if "mass" in e["past"]:
                    past.append(undo_resets(c.mass_kg.to_numpy(np.float32), reset_frac=0.3)[None])
                if "fill" in e["past"]:
                    past.append(undo_resets(c.fill_pct.to_numpy(np.float32), reset_drop=20.0)[None])
                ctx_l.append(c[target].to_numpy(np.float32)); act_l.append(a[target].to_numpy(np.float32))
                known_l.append(np.concatenate(known) if known else None)
                past_l.append(np.concatenate(past) if past else None)
                keys.append((i, cutoff))
            context = np.stack(ctx_l)[:, None, :]
            actual = np.stack(act_l)
            known_arr = np.stack(known_l) if known_l[0] is not None else None
            past_arr = np.stack(past_l) if past_l[0] is not None else None
            lead_hours = np.arange(1, H + 1, dtype=float)
            t0 = time.perf_counter()
            fc = get_model(e["model"]).forecast(context, H, known_arr, past_arr)
            print(f"lead {lead} d  {e['spec']:<28} {len(keys)} forecasts  {time.perf_counter() - t0:5.1f}s")
            for b, (i, cutoff) in enumerate(keys):
                ev = events.loc[i]
                thr = metas[ev.device_id].get("on_call_threshold_pct")
                thr = float(thr) if thr is not None else np.nan
                y = actual[b]
                ok = np.flatnonzero(np.isfinite(y))
                if not len(ok):
                    continue
                j = ok[-1]
                med = fc.q(0.5)[b, 0]
                lo = fc.q(0.1)[b, 0] if fc.has(0.1) else None
                hi = fc.q(0.9)[b, 0] if fc.has(0.9) else None
                rows.append({
                    "device_id": ev.device_id, "lift_at": ev.lift_at, "lead_days": lead, "experiment": e["spec"],
                    "policy": metas[ev.device_id].get("collection_policy"),
                    "prior_lift_in_window": bool(pd.notna(ev.prev_lift_at) and ev.prev_lift_at >= cutoff),
                    "mae_window": float(np.mean(np.abs(y[ok] - med[ok]))),
                    "fill_at_lift_actual": float(y[j]), "fill_at_lift_pred": float(med[j]),
                    "fill_at_lift_err": float(med[j] - y[j]),
                    "covered_80": bool(lo[j] <= y[j] <= hi[j]) if lo is not None else np.nan,
                    "event_mass_before": ev.event_mass_before, "event_collected_kg": ev.event_collected_kg,
                    "weighbridge_net_kg": ev.weighbridge_net_kg,
                    "width_80": float(hi[j] - lo[j]) if lo is not None else np.nan,
                    "hours_to_thr_actual": first_crossing_hours(y, thr, lead_hours) if np.isfinite(thr) else np.nan,
                    "hours_to_thr_median": first_crossing_hours(med, thr, lead_hours) if np.isfinite(thr) else np.nan,
                })

    res = pd.DataFrame(rows)
    res["abs_err_lift"] = res.fill_at_lift_err.abs()
    res["target"] = target
    if target == "mass_kg":
        res["rel_err"] = res.fill_at_lift_err / res.fill_at_lift_actual.clip(lower=1)
        res["abs_rel_err"] = res.rel_err.abs()
        res["wb_err"] = res.fill_at_lift_pred - res.weighbridge_net_kg
        res["wb_abs_rel_err"] = (res.wb_err / res.weighbridge_net_kg.clip(lower=1)).abs()
        res["sensor_vs_wb_abs_rel_err"] = ((res.fill_at_lift_actual - res.weighbridge_net_kg) / res.weighbridge_net_kg.clip(lower=1)).abs()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_event.csv", index=False)
    order = [e["spec"] for e in exps]
    for title, sub in (("all lifts", res), ("no other lift inside the window", res[~res.prior_lift_in_window])):
        s = sub.groupby(["lead_days", "experiment"]).agg(
            n=("abs_err_lift", "size"), mae_window=("mae_window", "mean"), mae_at_lift=("abs_err_lift", "mean"),
            median_abs_err=("abs_err_lift", "median"), bias_at_lift=("fill_at_lift_err", "mean"),
            coverage_80=("covered_80", "mean"), width_80=("width_80", "mean")).reset_index()
        s["experiment"] = pd.Categorical(s["experiment"], order)
        s = s.sort_values(["lead_days", "experiment"])
        print(f"\n== fill level before the lift, {title}")
        print(s.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
        if title == "all lifts":
            s.to_csv(out / "summary.csv", index=False)
    if target == "mass_kg":
        s = res.groupby(["lead_days", "experiment"]).agg(
            n=("abs_err_lift", "size"), mae_kg=("abs_err_lift", "mean"), median_abs_rel_err=("abs_rel_err", "median"),
            within_10pct=("abs_rel_err", lambda x: (x <= 0.10).mean()), within_20pct=("abs_rel_err", lambda x: (x <= 0.20).mean()),
            n_wb=("wb_abs_rel_err", "count"), mae_vs_wb_kg=("wb_err", lambda x: x.abs().mean()),
            median_rel_err_vs_wb=("wb_abs_rel_err", "median"), sensor_itself_vs_wb=("sensor_vs_wb_abs_rel_err", "median")).reset_index()
        s["experiment"] = pd.Categorical(s["experiment"], order)
        print("\n== mass before the lift: relative errors, and the forecast against the weighbridge ticket")
        print(s.sort_values(["lead_days", "experiment"]).to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        s.to_csv(out / "summary_mass.csv", index=False)
    # paired comparison against the first experiment
    base = order[0]
    piv = res.pivot_table(index=["device_id", "lift_at", "lead_days"], columns="experiment", values="abs_err_lift")
    print(f"\n== paired change in |error| at lift vs {base} (negative = better), mean and share of lifts improved")
    for lead in leads:
        pl = piv.xs(lead, level="lead_days")
        for ex in order[1:]:
            if ex in pl:
                d = (pl[ex] - pl[base]).dropna()
                print(f"  lead {lead} d  {ex:<28} mean {d.mean():+6.2f}   improved {100 * (d < 0).mean():5.1f} %   n={len(d)}")
    print(f"\nwrote {out}/per_event.csv and summary.csv")


if __name__ == "__main__":
    main()
