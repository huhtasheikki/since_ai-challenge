"""Scoring for one device's 15-day forecast, per lead-time bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import QuantileForecast

LEAD_BUCKETS = (("d1_3", 0, 72), ("d4_7", 72, 168), ("d8_15", 168, 10**9))


def lead_bucket(lead_hours: float) -> str:
    for name, lo, hi in LEAD_BUCKETS:
        if lo < lead_hours <= hi:
            return name
    return LEAD_BUCKETS[-1][0]


def pinball(y: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    diff = y - q
    return np.where(diff >= 0, level * diff, (level - 1) * diff)


def first_crossing_hours(values: np.ndarray, threshold: float, lead_hours: np.ndarray) -> float:
    """Lead time (h) of the first value >= threshold, NaN if never."""
    hit = np.flatnonzero(np.isfinite(values) & (values >= threshold))
    return float(lead_hours[hit[0]]) if len(hit) else np.nan


def score_forecast(
    actual: np.ndarray,            # (H,)
    fc: QuantileForecast,          # quantiles (H, Q) for this series (already indexed [b, v])
    lead_hours: np.ndarray,        # (H,)
    baselines: dict[str, np.ndarray] | None = None,  # name -> (H,) point forecasts
    threshold: float | None = None,
) -> pd.DataFrame:
    """One row per lead bucket plus an ``all`` row."""
    q = fc.quantiles
    med = fc.q(0.5)
    lo = fc.q(0.1) if fc.has(0.1) else None
    hi = fc.q(0.9) if fc.has(0.9) else None
    buckets = np.array([lead_bucket(h) for h in lead_hours])
    rows = []
    for name in [b[0] for b in LEAD_BUCKETS] + ["all"]:
        m = np.isfinite(actual) & ((buckets == name) if name != "all" else True)
        n = int(m.sum())
        if n == 0:
            continue
        y = actual[m]
        err = y - med[m]
        row = {
            "lead_bucket": name,
            "n": n,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(-err)),
            "crps": float(2 * np.mean([pinball(y, q[m, i], lv).mean() for i, lv in enumerate(fc.levels)]))
            if len(fc.levels) > 1 else np.nan,
            "coverage_80": float(np.mean((y >= lo[m]) & (y <= hi[m]))) if lo is not None else np.nan,
            "width_80": float(np.mean(hi[m] - lo[m])) if lo is not None else np.nan,
        }
        for bname, bpred in (baselines or {}).items():
            mae_b = float(np.mean(np.abs(y - bpred[m])))
            row[f"mae_{bname}"] = mae_b
            row[f"skill_vs_{bname}"] = 1 - row["mae"] / mae_b if mae_b > 0 else np.nan
        if name == "all" and threshold is not None:
            t_act = first_crossing_hours(actual, threshold, lead_hours)
            t_fc = first_crossing_hours(med, threshold, lead_hours)
            row["threshold_pct"] = threshold
            row["hours_to_threshold_actual"] = t_act
            row["hours_to_threshold_forecast"] = t_fc
            row["hours_to_threshold_error"] = t_fc - t_act
            if fc.has(0.9):  # upper quantile: "it could be full by then"
                t_q90 = first_crossing_hours(fc.q(0.9), threshold, lead_hours)
                row["hours_to_threshold_forecast_q90"] = t_q90
                row["hours_to_threshold_error_q90"] = t_q90 - t_act
        rows.append(row)
    return pd.DataFrame(rows)
