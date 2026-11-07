"""Write mass forecasts (next 96 h, hourly, q10/q50/q90) to <schema>.mass_forecast.

Default model: Timestral on the lift-undone mass series with 60 d context, the site calendar as
known covariate and lift-undone fill % + compaction cycles as past covariates, the best setup in
the backtests (README). Context is shortened automatically when less history exists.

    uv run python forecast_job.py --schema europress                      # issue at the latest data hour
    uv run python forecast_job.py --schema europress --issued-at 2026-11-05T05:00Z
    uv run python forecast_job.py --schema public --backfill-from 2026-08-01 --backfill-to 2026-09-27 --every-hours 24
    uv run python forecast_job.py --schema europress --dry-run            # forecast, print, write nothing

Tables are created with CREATE IF NOT EXISTS from fill_forecast/schema.sql; rows are upserted on
(device_id, model, issued_at, target_time), so re-running an issue time is safe.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import build_models
from covariate_test import parse_experiment
from fill_forecast import db
from fill_forecast.covariates import site_calendar_from_db, site_open_hours, undo_resets
from fill_forecast.models import set_value_range
from lift_backtest import utc

DEFAULT_MODEL = "timestral_cum@60+cal+fill+cycles"
SCHEMA_SQL = Path(__file__).parent / "fill_forecast" / "schema.sql"


def ensure_tables(conn, schema: str) -> None:
    sql = SCHEMA_SQL.read_text().replace("{schema}", schema)
    with conn.cursor() as cur:
        cur.execute(sql)


def floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--schema", default="europress", choices=db.SCHEMAS)
    p.add_argument("--model", default=DEFAULT_MODEL, help="experiment spec, see covariate_test.py")
    p.add_argument("--devices", help="comma-separated ids (default: all devices with a weight sensor)")
    p.add_argument("--issued-at", help="forecast origin (UTC hour). Default: latest measured hour in the schema")
    p.add_argument("--backfill-from"); p.add_argument("--backfill-to")
    p.add_argument("--every-hours", type=int, default=24)
    p.add_argument("--horizon-hours", type=int, default=96)
    p.add_argument("--min-context-days", type=int, default=7)
    p.add_argument("--timestral-repo", default="mistralai/Timestral")
    p.add_argument("--timesfm-checkpoint", default="google/timesfm-3.0-pytorch")
    p.add_argument("--device-type", default="cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)

    spec = parse_experiment(args.model)
    args.reset_frac = 0.3
    set_value_range(0.0, float("inf"))
    H = args.horizon_hours
    conn = db.connect()
    schema = args.schema

    # devices
    with conn.cursor() as cur:
        cur.execute(f"SELECT device_id, site_id, has_weight_measurement FROM {schema}.devices ORDER BY 1")
        dev_rows = cur.fetchall()
    devices = [d for d, _, w in dev_rows if w is not False]
    site_of = {d: s for d, s, _ in dev_rows}
    if args.devices:
        wanted = [d.strip() for d in args.devices.split(",")]
        devices = [d for d in devices if d in wanted]
    with conn.cursor() as cur:
        cur.execute(f"SELECT min(measured_at), max(measured_at) FROM {schema}.telemetry_hourly")
        data_min, data_max = cur.fetchone()
    cal_df = site_calendar_from_db(conn, schema)
    print(f"schema={schema}: {len(devices)} devices with a scale, data {data_min:%Y-%m-%d %H:%M} -> {data_max:%Y-%m-%d %H:%M}, "
          f"calendar from {'DB' if cal_df is not None else 'parquet'}")

    # issue times
    if args.backfill_from:
        t0, t1 = utc(args.backfill_from), utc(args.backfill_to or data_max.isoformat())
        issues = list(pd.date_range(t0, t1, freq=f"{args.every_hours}h", tz="UTC").to_pydatetime())
    else:
        issues = [utc(args.issued_at) if args.issued_at else floor_hour(data_max) + timedelta(hours=1)]  # end of last complete hour
    issues = [floor_hour(t).astimezone(timezone.utc) for t in issues]
    ctx_hours_full = spec["context_days"] * 24
    first_start = max(floor_hour(data_min) + timedelta(hours=1), issues[0] - timedelta(hours=ctx_hours_full))
    frames = {d: db.fetch_hourly_frame(conn, d, first_start, issues[-1], schema) for d in devices}
    model = build_models([spec["model"]], args)[spec["model"]]

    total_rows = 0
    for issued_at in issues:
        start = max(first_start, issued_at - timedelta(hours=ctx_hours_full))
        ctx_hours = int((issued_at - start).total_seconds() // 3600)
        if ctx_hours < args.min_context_days * 24:
            print(f"{issued_at:%Y-%m-%d %H:%M}: only {ctx_hours} h of context, skipped")
            continue
        horizon_idx = pd.date_range(pd.Timestamp(issued_at).tz_convert("UTC") + pd.Timedelta(hours=1), periods=H, freq="h")
        ctx_l, known_l, past_l, used = [], [], [], []
        for d in devices:
            f = frames[d]
            c = f[(f.index >= start) & (f.index < issued_at)]
            # usable: full-length window, at least half of the last 7 days valid, something valid in the last 24 h
            if len(c) != ctx_hours or c.mass_kg.tail(168).notna().mean() < 0.5 or c.mass_kg.tail(24).notna().sum() == 0:
                continue
            full_idx = c.index.append(horizon_idx)
            known = []
            if "cal" in spec["known"]:
                known.append(site_open_hours(site_of.get(d), full_idx, cal_df))
            past = []
            if "cycles" in spec["past"]:
                past.append(c.compaction_cycles.fillna(0).to_numpy(np.float32)[None])
            if "fill" in spec["past"]:
                past.append(undo_resets(c.fill_pct.to_numpy(np.float32), reset_drop=20.0)[None])
            ctx_l.append(c.mass_kg.to_numpy(np.float32)); used.append(d)
            known_l.append(np.concatenate(known) if known else None)
            past_l.append(np.concatenate(past) if past else None)
        if not used:
            print(f"{issued_at:%Y-%m-%d %H:%M}: no device with usable context, skipped")
            continue
        skipped = sorted(set(devices) - set(used))
        if skipped and len(issues) == 1:
            print(f"  skipped (no usable mass in context): {', '.join(skipped)}")
        context = np.stack(ctx_l)[:, None, :]
        known_arr = np.stack(known_l) if known_l[0] is not None else None
        past_arr = np.stack(past_l) if past_l[0] is not None else None
        t_start = time.perf_counter()
        fc = model.forecast(context, H, known_arr, past_arr)
        seconds = time.perf_counter() - t_start
        last_mass = np.array([c[np.isfinite(c)][-1] for c in context[:, 0, :]], dtype=np.float32)
        q10, q50, q90 = fc.q(0.1)[:, 0], fc.q(0.5)[:, 0], fc.q(0.9)[:, 0]
        print(f"{issued_at:%Y-%m-%d %H:%M}: {len(used)} devices, {ctx_hours} h context, {seconds:.1f}s, "
              f"median mass now {np.median(last_mass):.0f} kg -> +48 h {np.median(q50[:, 47]):.0f} kg -> +96 h {np.median(q50[:, -1]):.0f} kg")
        if args.dry_run:
            continue
        rows = []
        for b, d in enumerate(used):
            for k in range(H):
                rows.append((d, args.model, issued_at, horizon_idx[k].to_pydatetime(), k + 1,
                             float(q10[b, k]), float(q50[b, k]), float(q90[b, k]), float(last_mass[b])))
        ensure_tables(conn, schema)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {schema}.mass_forecast_run (issued_at, model, model_revision, context_hours, horizon_hours, n_devices, seconds, notes) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id",
                    (issued_at, args.model, getattr(model, "revision", None), ctx_hours, H, len(used), seconds, args.notes or None))
                run_id = cur.fetchone()[0]
                cur.executemany(
                    f"INSERT INTO {schema}.mass_forecast (device_id, model, issued_at, target_time, lead_hours, "
                    f"mass_kg_q10, mass_kg_q50, mass_kg_q90, mass_kg_at_issue, run_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, {run_id}) "
                    f"ON CONFLICT (device_id, model, issued_at, target_time) DO UPDATE SET mass_kg_q10 = EXCLUDED.mass_kg_q10, "
                    f"mass_kg_q50 = EXCLUDED.mass_kg_q50, mass_kg_q90 = EXCLUDED.mass_kg_q90, mass_kg_at_issue = EXCLUDED.mass_kg_at_issue, "
                    f"run_id = EXCLUDED.run_id", rows)
        total_rows += len(rows)
    print(f"{'dry run, nothing written' if args.dry_run else f'wrote {total_rows} rows to {schema}.mass_forecast'}")


if __name__ == "__main__":
    main()
