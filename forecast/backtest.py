"""Backtest: 30 days of fill_pct -> 15-day forecast per device, TimesFM 3.0 vs Timestral vs baselines.

Example:
    uv run python backtest.py --devices CMP-00013,CMP-00007 --cutoff 2026-08-16
    uv run python backtest.py --device CMP-00013 --models persistence,ramp   # smoke test, no weights
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fill_forecast import db
from fill_forecast.metrics import score_forecast
from fill_forecast.models import Persistence, QuantileForecast, Ramp, hour_of_week_covariates
from fill_forecast.plot import plot_run

BASELINES = ("persistence", "ramp")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", help="single device id, e.g. CMP-00013")
    p.add_argument("--devices", help="comma-separated device ids (batched through each model)")
    p.add_argument("--cutoff", help="forecast origin, ISO UTC (default: latest data minus horizon)")
    p.add_argument("--context-days", type=int, default=30)
    p.add_argument("--horizon-days", type=int, default=15)
    p.add_argument("--resolution", type=int, default=1, help="bucket size in hours (1 or 3)")
    p.add_argument("--models", default="timestral,timesfm3", help="comma list of timestral,timesfm3,persistence,ramp")
    p.add_argument("--timestral-repo", default="mistralai/Timestral")
    p.add_argument("--timesfm-checkpoint", default="google/timesfm-3.0-pytorch")
    p.add_argument("--device-type", default="cpu", help="torch device: cpu, mps, cuda")
    p.add_argument("--schema", default="public", choices=db.SCHEMAS)
    p.add_argument("--as-of", action="store_true", help="hide rows received after the cutoff from the context")
    p.add_argument("--known-covariates", action="store_true", help="feed hour-of-week covariates to the models")
    p.add_argument("--out", default="outputs")
    return p.parse_args(argv)


def build_models(names: list[str], args) -> dict:
    models = {}
    for n in names:
        if n == "timestral":
            from fill_forecast.models import Timestral
            models[n] = Timestral(repo=args.timestral_repo, device=args.device_type)
        elif n == "timesfm3":
            from fill_forecast.models import TimesFM3
            models[n] = TimesFM3(checkpoint=args.timesfm_checkpoint, device=args.device_type)
        elif n in ("timestral_cum", "timesfm3_cum"):
            from fill_forecast.models import Cumulative
            models[n] = Cumulative(build_models([n[:-4]], args)[n[:-4]], reset_frac=getattr(args, "reset_frac", None))
        elif n == "persistence":
            models[n] = Persistence()
        elif n == "ramp":
            models[n] = Ramp(reset_frac=getattr(args, "reset_frac", None))
        else:
            sys.exit(f"unknown model {n}")
    return models


def main(argv=None):
    args = parse_args(argv)
    devices = [d.strip() for d in (args.devices or args.device or "CMP-00013").split(",") if d.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for b in BASELINES:  # baselines are always computed for skill scores
        if b not in model_names:
            model_names.append(b)

    conn = db.connect()
    res = args.resolution
    ctx_n = args.context_days * 24 // res
    hor_n = args.horizon_days * 24 // res

    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff).tz_localize("UTC") if pd.Timestamp(args.cutoff).tzinfo is None else pd.Timestamp(args.cutoff).tz_convert("UTC")
        cutoff = cutoff.to_pydatetime()
    else:
        latest = min(db.latest_measured_at(conn, d, args.schema) for d in devices)
        cutoff = (latest - timedelta(days=args.horizon_days)).replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    start, end = cutoff - timedelta(days=args.context_days), cutoff + timedelta(days=args.horizon_days)
    print(f"cutoff {cutoff.isoformat()}  context {start.date()} -> {cutoff.date()}  horizon -> {end.date()}  schema={args.schema}")

    # ---- fetch ----
    series, metas, events = {}, {}, {}
    for d in devices:
        full = db.fetch_fill_series(conn, d, start, end, args.schema, resolution_hours=res)
        if args.as_of:
            ctx = db.fetch_fill_series(conn, d, start, cutoff, args.schema, as_of=cutoff, resolution_hours=res)
            full = pd.concat([ctx, full[full.index >= cutoff]])
        if len(full) != ctx_n + hor_n:
            sys.exit(f"{d}: expected {ctx_n + hor_n} buckets, got {len(full)}")
        series[d] = full
        metas[d] = db.fetch_device_meta(conn, d, args.schema)
        events[d] = db.fetch_collection_events(conn, d, start, end, args.schema)
        ctx_nan = int(full.iloc[:ctx_n].isna().sum())
        print(f"  {d}: {metas[d].get('collection_policy')} thr={metas[d].get('on_call_threshold_pct')} "
              f"{metas[d].get('material_stream')} {metas[d].get('container_volume_m3')} m³, "
              f"context NaN {ctx_nan}/{ctx_n}, lifts in window {len(events[d])}")

    context = np.stack([series[d].values[:ctx_n] for d in devices])[:, None, :].astype(np.float32)  # (B,1,T)
    actual = np.stack([series[d].values[ctx_n:] for d in devices]).astype(np.float32)               # (B,H)
    lead_hours = (np.arange(1, hor_n + 1) * res).astype(float)
    cov = None
    if args.known_covariates:
        s0 = series[devices[0]]
        cov = np.stack([hour_of_week_covariates(s0.index[:ctx_n], s0.index[ctx_n:])] * len(devices))

    # ---- forecast ----
    models = build_models(model_names, args)
    forecasts: dict[str, QuantileForecast] = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        forecasts[name] = m.forecast(context, hor_n, cov)
        print(f"  {name}: {time.perf_counter() - t0:.1f}s for {len(devices)} device(s)")

    # ---- score, save ----
    out_root = Path(args.out)
    all_metrics = []
    for b, d in enumerate(devices):
        thr = metas[d].get("on_call_threshold_pct")
        thr = float(thr) if thr is not None else None
        base_pts = {bn: forecasts[bn].q(0.5)[b, 0] for bn in BASELINES}
        per_dev = {}
        rows = []
        for name, fc in forecasts.items():
            fc_b = QuantileForecast(fc.quantiles[b, 0], fc.levels, fc.seconds, fc.revision)
            per_dev[name] = fc_b
            df = score_forecast(actual[b], fc_b, lead_hours, baselines=base_pts, threshold=thr)
            df.insert(0, "model", name)
            df.insert(0, "device_id", d)
            rows.append(df)
        m = pd.concat(rows, ignore_index=True)
        all_metrics.append(m)

        run_dir = out_root / f"{d}_{cutoff.strftime('%Y%m%dT%H')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        fcsv = pd.DataFrame({"target_time": series[d].index[ctx_n:], "lead_hours": lead_hours, "actual": actual[b]})
        for name, fc_b in per_dev.items():
            fcsv[f"{name}_q50"] = fc_b.q(0.5)
            if fc_b.has(0.1):
                fcsv[f"{name}_q10"] = fc_b.q(0.1)
                fcsv[f"{name}_q90"] = fc_b.q(0.9)
        fcsv.to_csv(run_dir / "forecast.csv", index=False)
        m.to_csv(run_dir / "metrics.csv", index=False)
        (run_dir / "run.json").write_text(json.dumps({
            "device_id": d, "cutoff": cutoff.isoformat(), "context_days": args.context_days,
            "horizon_days": args.horizon_days, "resolution_hours": res, "schema": args.schema,
            "as_of": args.as_of, "known_covariates": args.known_covariates, "meta": metas[d],
            "models": {n: {"revision": fc.revision, "batch_seconds": fc.seconds, "levels": list(fc.levels)}
                       for n, fc in forecasts.items()},
        }, indent=2, default=str))
        plot_run(str(run_dir / "plot.png"), d, series[d].iloc[:ctx_n], series[d].iloc[ctx_n:],
                 {n: per_dev[n] for n in model_names}, m, events[d], thr, metas[d])
        print(f"\n== {d}  ({run_dir})")
        cols = ["model", "lead_bucket", "n", "mae", "rmse", "bias", "crps", "coverage_80",
                "skill_vs_persistence", "skill_vs_ramp"]
        print(m[cols].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
        if thr is not None:
            cols_t = [c for c in m.columns if c.startswith("hours_to_threshold")]
            ttc = m[m.lead_bucket == "all"][["model", *cols_t]]
            print(ttc.to_string(index=False))

    summary = pd.concat(all_metrics, ignore_index=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    summary.to_csv(out_root / f"summary_{stamp}.csv", index=False)
    if len(devices) > 1:
        print("\n== mean over devices (MAE by lead bucket)")
        piv = summary.pivot_table(index="model", columns="lead_bucket", values="mae", aggfunc="mean")
        print(piv.reindex(model_names).to_string(float_format=lambda x: f"{x:6.2f}"))
        piv2 = summary.pivot_table(index="model", columns="lead_bucket", values="coverage_80", aggfunc="mean")
        print("\n== mean coverage_80")
        print(piv2.reindex(model_names).to_string(float_format=lambda x: f"{x:6.2f}"))


if __name__ == "__main__":
    main()
