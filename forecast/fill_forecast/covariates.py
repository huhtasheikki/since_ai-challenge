"""Extra inputs for the forecasters: site opening calendar (known future) and hour-of-week."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Challenge data pack unzipped next to this project (override with CHALLENGE_DATA_DIR).
DEFAULT_DATA_DIR = (Path(__file__).resolve().parents[2] / "Europress_SinceAI2026_FINAL"
                    / "Europress_SMART_Material_Value_Engine_v1.0.1" / "data" / "competition" / "public")


def data_dir() -> Path:
    return Path(os.environ.get("CHALLENGE_DATA_DIR", DEFAULT_DATA_DIR))


@lru_cache(maxsize=1)
def site_calendar() -> pd.DataFrame:
    cal = pd.read_parquet(data_dir() / "reference" / "site_calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return cal.set_index(["site_id", "date"]).sort_index()


@lru_cache(maxsize=1)
def sites() -> pd.DataFrame:
    return pd.read_parquet(data_dir() / "reference" / "sites.parquet").set_index("site_id")


def site_calendar_from_db(conn, schema: str) -> pd.DataFrame | None:
    """Same shape as site_calendar() but read from <schema>.site_calendar; None if absent."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = 'site_calendar'", (schema,))
        if cur.fetchone() is None:
            return None
        cur.execute(f"SELECT site_id, date, is_open, open_hour_local, close_hour_local FROM {schema}.site_calendar")
        rows = cur.fetchall()
    cal = pd.DataFrame(rows, columns=["site_id", "date", "is_open", "open_hour_local", "close_hour_local"])
    cal["date"] = pd.to_datetime(cal["date"])
    return cal.set_index(["site_id", "date"]).sort_index()


def site_open_hours(site_id: str, index: pd.DatetimeIndex, cal_df: pd.DataFrame | None = None,
                    tz: str = "Europe/Helsinki") -> np.ndarray:
    """(2, T) known covariates: site open this hour (0/1), site open this day (0/1).

    Uses the site's local timezone; a day missing from the calendar counts as open.
    ``cal_df`` defaults to the challenge parquet calendar.
    """
    if cal_df is None:
        cal_df = site_calendar()
        if site_id in sites().index:
            tz = sites().loc[site_id, "timezone"]
    local = index.tz_convert(tz)
    days = pd.DatetimeIndex(local.normalize().tz_localize(None))
    try:
        cal = cal_df.loc[site_id]
    except KeyError:
        return np.ones((2, len(index)), dtype=np.float32)
    cal = cal.reindex(days)
    is_open = cal["is_open"].fillna(True).to_numpy(dtype=bool)
    oh = cal["open_hour_local"].fillna(0).to_numpy(dtype=float)
    ch = cal["close_hour_local"].fillna(24).to_numpy(dtype=float)
    h = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    open_hour = is_open & (h >= oh) & (h < ch)
    return np.stack([open_hour, is_open]).astype(np.float32)


def hour_of_week(index: pd.DatetimeIndex, tz: str = "Europe/Helsinki") -> np.ndarray:
    """(5, T): sin/cos hour-of-day, sin/cos day-of-week, weekend flag (local time)."""
    local = index.tz_convert(tz)
    h = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    d = local.dayofweek.to_numpy()
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
                     (d >= 5).astype(float)]).astype(np.float32)


def undo_resets(x: np.ndarray, reset_drop: float = 20.0, reset_frac: float | None = None) -> np.ndarray:
    """Cumulative version of a 1-D series with big drops removed (NaN-safe via interpolation)."""
    from .models import _fill_gaps
    filled = _fill_gaps(x[None, None, :])[0, 0]
    d = np.diff(filled)
    thr = reset_frac * np.percentile(filled, 95) if reset_frac else reset_drop
    d = np.where(d < -thr, 0.0, d)
    return np.concatenate([filled[:1], filled[:1] + np.cumsum(d)]).astype(np.float32)
