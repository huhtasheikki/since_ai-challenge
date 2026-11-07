"""Read-only access to the Tiger Cloud (TimescaleDB) copy of the Europress challenge data.

Two schemas hold the same tables: ``public`` (25-device year-long CSV sample) and
``europress`` (Torsti's live simulated feed). ``telemetry_hourly`` is a raw ingest log, so
rows are de-duplicated on ``message_id`` (earliest ``received_at`` wins) before bucketing.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

SCHEMAS = ("public", "europress")


def _schema(schema: str) -> str:
    if schema not in SCHEMAS:
        raise ValueError(f"schema must be one of {SCHEMAS}, got {schema!r}")
    return schema


def connect() -> psycopg.Connection:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (copy .env.example to .env)")
    return psycopg.connect(url, autocommit=True)


def latest_measured_at(conn: psycopg.Connection, device_id: str, schema: str = "public") -> datetime:
    s = _schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT max(measured_at) FROM {s}.telemetry_hourly WHERE device_id = %s", (device_id,))
        row = cur.fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"no telemetry for device {device_id} in schema {schema}")
    return row[0]


def fetch_fill_series(
    conn: psycopg.Connection,
    device_id: str,
    start: datetime,
    end: datetime,
    schema: str = "public",
    as_of: datetime | None = None,
    resolution_hours: int = 1,
) -> pd.Series:
    """Hourly (or ``resolution_hours``) mean ``fill_pct`` on a regular grid in [start, end).

    Missing buckets are NaN. ``as_of`` drops rows received after that instant so a backtest
    context cannot see buffered/late deliveries that were not available at the cutoff.
    """
    s = _schema(schema)
    as_of_clause = "AND received_at <= %(as_of)s" if as_of is not None else ""
    sql = f"""
        WITH dedup AS (
            SELECT DISTINCT ON (message_id) message_id, measured_at, fill_pct
            FROM {s}.telemetry_hourly
            WHERE device_id = %(device_id)s
              AND measured_at >= %(start)s AND measured_at < %(end)s
              {as_of_clause}
            ORDER BY message_id, received_at, delivery_id
        )
        SELECT time_bucket_gapfill(%(bucket)s::interval, measured_at, %(start)s::timestamptz, %(end)s::timestamptz) AS bucket,
               avg(fill_pct) AS fill_pct
        FROM dedup
        GROUP BY bucket
        ORDER BY bucket
    """
    params = {
        "device_id": device_id,
        "start": start,
        "end": end,
        "as_of": as_of,
        "bucket": f"{resolution_hours} hours",
    }
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC", name="bucket")
    vals = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=np.float32)
    return pd.Series(vals, index=idx, name="fill_pct")


def fetch_device_meta(conn: psycopg.Connection, device_id: str, schema: str = "public") -> dict:
    s = _schema(schema)
    cols = (
        "device_id, site_id, customer_id, equipment_class, container_volume_m3, material_stream, "
        "collection_policy, fixed_schedule_per_week, fixed_schedule_weekdays, on_call_threshold_pct, "
        "hybrid_skip_below_pct, operator_id"
    )
    with conn.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM {s}.devices WHERE device_id = %s", (device_id,))
        row = cur.fetchone()
        names = [d.name for d in cur.description]
    if row is None:
        return {"device_id": device_id}
    return dict(zip(names, row))


def fetch_collection_events(
    conn: psycopg.Connection, device_id: str, start: datetime, end: datetime, schema: str = "public"
) -> pd.DataFrame:
    s = _schema(schema)
    sql = f"""
        SELECT e.actual_at, e.trigger_type, e.status, e.device_fill_pct_before, e.device_fill_pct_after,
               e.device_mass_kg_before, e.device_mass_kg_after, e.device_estimated_collected_kg,
               w.allocated_net_kg AS weighbridge_net_kg
        FROM {s}.collection_events e
        LEFT JOIN {s}.weighbridge_tickets w USING (operator_job_ref)
        WHERE e.device_id = %s AND e.actual_at >= %s AND e.actual_at < %s
        ORDER BY e.actual_at
    """
    with conn.cursor() as cur:
        cur.execute(sql, (device_id, start, end))
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    df = pd.DataFrame(rows, columns=names)
    if not df.empty:
        df["actual_at"] = pd.to_datetime(df["actual_at"], utc=True)
    return df


def fetch_hourly_frame(
    conn: psycopg.Connection,
    device_id: str,
    start: datetime,
    end: datetime,
    schema: str = "public",
) -> pd.DataFrame:
    """Hourly grid in [start, end) with fill_pct, compaction_cycles and mass_kg (NaN for gaps)."""
    s = _schema(schema)
    sql = f"""
        WITH dedup AS (
            SELECT DISTINCT ON (message_id) message_id, measured_at, fill_pct, compaction_cycles, mass_kg, quality_flag
            FROM {s}.telemetry_hourly
            WHERE device_id = %(device_id)s AND measured_at >= %(start)s AND measured_at < %(end)s
            ORDER BY message_id, received_at, delivery_id
        )
        SELECT time_bucket_gapfill('1 hour'::interval, measured_at, %(start)s::timestamptz, %(end)s::timestamptz) AS bucket,
               avg(fill_pct) AS fill_pct,
               sum(compaction_cycles) AS compaction_cycles,
               avg(mass_kg) FILTER (WHERE quality_flag = 'VALID') AS mass_kg
        FROM dedup GROUP BY bucket ORDER BY bucket
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"device_id": device_id, "start": start, "end": end})
        rows = cur.fetchall()
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC", name="bucket")
    df = pd.DataFrame({
        "fill_pct": [np.nan if r[1] is None else float(r[1]) for r in rows],
        "compaction_cycles": [np.nan if r[2] is None else float(r[2]) for r in rows],
        "mass_kg": [np.nan if r[3] is None else float(r[3]) for r in rows],
    }, index=idx, dtype=np.float32)
    return df
