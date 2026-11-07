"""Event-anchored backtest: how well is the fill level before a lift predicted x days earlier?

For every COMPLETED/PARTIAL collection event, a forecast is issued at lift_hour - x days
(x in --leads) with 30 days of context and a horizon that ends at the last hour before the
lift. Scored per (event, lead, model):
  * mae_window: MAE over the whole pre-lift window
  * fill_at_lift_*: predicted (median / q10 / q90) vs actual fill in the last hour before the lift
  * threshold crossing (ON_CALL devices): hours until median / q90 reach the on-call threshold

Example:
    uv run python lift_backtest.py --from 2026-06-01 --to 2026-10-01 --leads 2,4,5,7
    uv run python lift_backtest.py --devices CMP-00013,CMP-00007 --models persistence,ramp
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fill_forecast import db
from fill_forecast.metrics import first_crossing_hours
from fill_forecast.models import QuantileForecast
from backtest import build_models, BASELINES

RESET_STATUSES = ("COMPLETED", "PARTIAL")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", help="comma-separated ids (default: every device in the schema)")
    p.add_argument("--from", dest="date_from", default="2026-06-01", help="first lift date (UTC)")
    p.add_argument("--to", dest="date_to", default="2026-10-01", help="exclusive last lift date (UTC)")
    p.add_argument("--leads", default="2,4,5,7", help="days before the lift at which to forecast")
    p.add_argument("--context-days", type=int, default=30)
    p.add_argument("--models", default="timestral,timesfm3")
    p.add_argument("--timestral-repo", default="mistralai/Timestral")
    p.add_argument("--timesfm-checkpoint", default="google/timesfm-3.0-pytorch")
    p.add_argument("--device-type", default="cpu")
    p.add_argument("--schema", default="public", choices=db.SCHEMAS)
    p.add_argument("--max-events", type=int, default=0, help="limit for quick runs (0 = all)")
    p.add_argument("--examples", type=int, default=3, help="number of example event plots")
    p.add_argument("--out", default="outputs/lift_backtest")
    return p.parse_args(argv)


def utc(s: str) -> datetime:
    t = pd.Timestamp(s)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


def main(argv=None):
    args = parse_args(argv)
    leads = [int(x) for x in args.leads.split(",")]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for b in BASELINES:
        if b not in model_names:
            model_names.append(b)
    ctx_n = args.context_days * 24
    date_from, date_to = utc(args.date_from), utc(args.date_to)
    conn = db.connect()

    # ---- devices + lift events ----
    with conn.cursor() as cur:
        if args.devices:
            devices = [d.strip() for d in args.devices.split(",")]
        else:
            cur.execute(f"SELECT device_id FROM {args.schema}.devices ORDER BY 1")
            devices = [r[0] for r in cur.fetchall()]
    metas = {d: db.fetch_device_meta(conn, d, args.schema) for d in devices}
    events = []
    for d in devices:
        ev = db.fetch_collection_events(conn, d, date_from, date_to, args.schema)
        ev = ev[ev["status"].isin(RESET_STATUSES)]
        for _, r in ev.iterrows():
            events.append({"device_id": d, "lift_at": r["actual_at"], "trigger_type": r["trigger_type"],
                           "status": r["status"], "fill_before_event": r["device_fill_pct_before"],
                           "fill_after_event": r["device_fill_pct_after"]})
    events = pd.DataFrame(events).sort_values(["device_id", "lift_at"]).reset_index(drop=True)
    if args.max_events:
        events = events.head(args.max_events)
    print(f"{len(events)} lifts on {events.device_id.nunique()} devices, leads {leads} days, schema={args.schema}")

    # ---- one long hourly series per device, sliced locally ----
    series_start = date_from - timedelta(days=args.context_days + max(leads) + 1)
    series = {}
    t0 = time.perf_counter()
    for d in devices:
        series[d] = db.fetch_fill_series(conn, d, series_start, date_to, args.schema)
    print(f"fetched {len(devices)} series in {time.perf_counter() - t0:.1f}s")

    # previous lift per event, to flag windows that contain another lift
    events["prev_lift_at"] = events.groupby("device_id")["lift_at"].shift(1)

    models = build_models(model_names, args)
    rows = []
    example_store = {}  # (device, lift_at) -> {lead: {model: (index, q10, q50, q90)}}
    for lead in leads:
        H = lead * 24
        ctx_list, act_list, idx_list, keys = [], [], [], []
        for i, ev in events.iterrows():
            s = series[ev.device_id]
            lift_hour = ev.lift_at.floor("h")
            cutoff = lift_hour - pd.Timedelta(hours=H)
            start = cutoff - pd.Timedelta(hours=ctx_n)
            ctx = s[(s.index >= start) & (s.index < cutoff)]
            act = s[(s.index >= cutoff) & (s.index < lift_hour)]
            if len(ctx) != ctx_n or len(act) != H or ctx.isna().mean() > 0.3:
                continue
            ctx_list.append(ctx.values); act_list.append(act.values); idx_list.append(act.index)
            keys.append((i, cutoff, lift_hour))
        if not keys:
            continue
        context = np.stack(ctx_list)[:, None, :].astype(np.float32)
        actual = np.stack(act_list).astype(np.float32)
        lead_hours = np.arange(1, H + 1, dtype=float)
        print(f"lead {lead} d: {len(keys)} forecasts, horizon {H} h")
        fcs: dict[str, QuantileForecast] = {}
        for name, m in models.items():
            t0 = time.perf_counter()
            fcs[name] = m.forecast(context, H, None)
            print(f"   {name}: {time.perf_counter() - t0:.1f}s")
        for b, (i, cutoff, lift_hour) in enumerate(keys):
            ev = events.loc[i]
            thr = metas[ev.device_id].get("on_call_threshold_pct")
            thr = float(thr) if thr is not None else np.nan
            y = actual[b]
            last_ok = np.flatnonzero(np.isfinite(y))
            y_lift = float(y[last_ok[-1]]) if len(last_ok) else np.nan
            prior_lift_in_window = bool(pd.notna(ev.prev_lift_at) and ev.prev_lift_at >= cutoff)
            for name, fc in fcs.items():
                med = fc.q(0.5)[b, 0]
                lo = fc.q(0.1)[b, 0] if fc.has(0.1) else None
                hi = fc.q(0.9)[b, 0] if fc.has(0.9) else None
                ok = np.isfinite(y)
                j = last_ok[-1] if len(last_ok) else H - 1
                row = {
                    "device_id": ev.device_id, "lift_at": ev.lift_at, "trigger_type": ev.trigger_type,
                    "policy": metas[ev.device_id].get("collection_policy"), "threshold_pct": thr,
                    "lead_days": lead, "cutoff": cutoff, "model": name,
                    "prior_lift_in_window": prior_lift_in_window,
                    "fill_at_cutoff": float(np.nan_to_num(context[b, 0, -1], nan=np.nan)),
                    "mae_window": float(np.mean(np.abs(y[ok] - med[ok]))) if ok.any() else np.nan,
                    "fill_at_lift_actual": y_lift,
                    "fill_before_event": ev.fill_before_event,
                    "fill_at_lift_pred": float(med[j]),
                    "fill_at_lift_err": float(med[j]) - y_lift,
                    "fill_at_lift_q10": float(lo[j]) if lo is not None else np.nan,
                    "fill_at_lift_q90": float(hi[j]) if hi is not None else np.nan,
                    "covered_80": bool(lo[j] <= y_lift <= hi[j]) if lo is not None else np.nan,
                    "hours_to_thr_actual": first_crossing_hours(y, thr, lead_hours) if np.isfinite(thr) else np.nan,
                    "hours_to_thr_median": first_crossing_hours(med, thr, lead_hours) if np.isfinite(thr) else np.nan,
                    "hours_to_thr_q90": first_crossing_hours(hi, thr, lead_hours) if (hi is not None and np.isfinite(thr)) else np.nan,
                }
                rows.append(row)
                key = (ev.device_id, ev.lift_at)
                if len(example_store) < args.examples or key in example_store:
                    example_store.setdefault(key, {}).setdefault(lead, {})[name] = (idx_list[b], lo, med, hi)

    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_event.csv", index=False)

    # ---- summaries ----
    res["abs_err_lift"] = res.fill_at_lift_err.abs()
    g = res.groupby(["lead_days", "model"])
    summary = g.agg(n=("abs_err_lift", "size"),
                    mae_window=("mae_window", "mean"),
                    mae_at_lift=("abs_err_lift", "mean"),
                    bias_at_lift=("fill_at_lift_err", "mean"),
                    coverage_80_at_lift=("covered_80", "mean"),
                    mean_actual_at_lift=("fill_at_lift_actual", "mean")).reset_index()
    summary["model"] = pd.Categorical(summary["model"], model_names)
    summary = summary.sort_values(["lead_days", "model"])
    summary.to_csv(out / "summary.csv", index=False)
    print("\n== fill level just before the lift: forecast issued N days earlier (all lifts)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    clean = res[~res.prior_lift_in_window]
    s2 = clean.groupby(["lead_days", "model"]).agg(n=("abs_err_lift", "size"), mae_window=("mae_window", "mean"),
                                                   mae_at_lift=("abs_err_lift", "mean"), bias_at_lift=("fill_at_lift_err", "mean"),
                                                   coverage_80_at_lift=("covered_80", "mean")).reset_index()
    s2["model"] = pd.Categorical(s2["model"], model_names)
    print("\n== same, only windows with no other lift between cutoff and this lift")
    print(s2.sort_values(["lead_days", "model"]).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    oc = res[res.policy == "ON_CALL"].copy()
    if not oc.empty:
        oc["thr_reached_actual"] = oc.hours_to_thr_actual.notna()
        oc["thr_reached_median"] = oc.hours_to_thr_median.notna()
        oc["thr_reached_q90"] = oc.hours_to_thr_q90.notna()
        oc["err_h_median"] = oc.hours_to_thr_median - oc.hours_to_thr_actual
        oc["err_h_q90"] = oc.hours_to_thr_q90 - oc.hours_to_thr_actual
        s3 = oc.groupby(["lead_days", "model"]).agg(
            n=("thr_reached_actual", "size"), actual_reached=("thr_reached_actual", "mean"),
            median_reached=("thr_reached_median", "mean"), q90_reached=("thr_reached_q90", "mean"),
            median_err_h=("err_h_median", "mean"), median_abs_err_h=("err_h_median", lambda x: x.abs().mean()),
            q90_err_h=("err_h_q90", "mean"), q90_abs_err_h=("err_h_q90", lambda x: x.abs().mean())).reset_index()
        s3["model"] = pd.Categorical(s3["model"], model_names)
        print("\n== ON_CALL devices: did the forecast reach the on-call threshold before the lift? (share, and timing error in hours)")
        print(s3.sort_values(["lead_days", "model"]).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
        s3.to_csv(out / "summary_threshold.csv", index=False)

    # ---- plots ----
    plot_summary(out / "summary.png", res, model_names, leads)
    for k, (key, per_lead) in enumerate(example_store.items()):
        plot_example(out / f"example_{k + 1}_{key[0]}.png", key, per_lead, series[key[0]], metas[key[0]], model_names, leads)
    print(f"\nwrote {out}/per_event.csv, summary.csv, summary.png and {len(example_store)} example plots")


def plot_summary(path, res, model_names, leads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fill_forecast.plot import SERIES_COLORS, INK, INK2, GRID

    fig, axes = plt.subplots(1, len(leads), figsize=(4.2 * len(leads), 4.4), dpi=140, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    fm = [m for m in model_names if m not in BASELINES]
    for ax, lead in zip(np.atleast_1d(axes), leads):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 100], [0, 100], color=GRID, lw=1)
        for i, name in enumerate(fm):
            r = res[(res.lead_days == lead) & (res.model == name)]
            ax.scatter(r.fill_at_lift_actual, r.fill_at_lift_pred, s=14, alpha=0.6, color=SERIES_COLORS[i],
                       edgecolor="none", label=f"{name}  MAE {r.abs_err_lift.mean():.1f}")
        ax.set_title(f"forecast issued {lead} d before lift", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("actual fill % before lift", color=INK2)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("forecast median at lift hour", color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_example(path, key, per_lead, s, meta, model_names, leads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from fill_forecast.plot import SERIES_COLORS, INK, INK2, GRID

    device_id, lift_at = key
    fm = [m for m in model_names if m not in BASELINES]
    fig, axes = plt.subplots(len(fm), 1, figsize=(13, 3.6 * len(fm)), dpi=140, sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    lift_hour = lift_at.floor("h")
    lo_t, hi_t = lift_hour - pd.Timedelta(days=max(leads) + 5), lift_hour + pd.Timedelta(days=2)
    seg = s[(s.index >= lo_t) & (s.index <= hi_t)]
    thr = meta.get("on_call_threshold_pct")
    for ax, name in zip(axes[:, 0], fm):
        ax.set_facecolor("#fcfcfb")
        ax.plot(seg.index, seg.values, color=INK, lw=1.4, label="actual")
        for i, lead in enumerate(leads):
            if lead not in per_lead or name not in per_lead[lead]:
                continue
            idx, lo, med, hi = per_lead[lead][name]
            c = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax.plot(idx, med, color=c, lw=1.8, label=f"issued {lead} d before")
            if lo is not None:
                ax.fill_between(idx, lo, hi, color=c, alpha=0.10, lw=0)
            ax.axvline(idx[0], color=c, lw=0.8, ls=":")
        ax.axvline(lift_hour, color="#008300", lw=1)
        if thr is not None:
            ax.axhline(thr, color="#e34948", lw=1, ls=":", alpha=0.8)
        ax.set_title(f"{device_id} · {meta.get('collection_policy')} · {meta.get('material_stream')} · lift {lift_at:%Y-%m-%d %H:%M} · {name}",
                     loc="left", fontsize=10, color=INK)
        ax.set_ylim(-2, 108); ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
