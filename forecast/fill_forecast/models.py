"""Forecasters with one output type. Context is (B, V, T) float32 with NaN for gaps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

FILL_MIN, FILL_MAX = 0.0, 100.0
_VALUE_RANGE = [FILL_MIN, FILL_MAX]


def set_value_range(lo: float, hi: float) -> None:
    """Clip range applied to every forecast (0..100 for fill_pct; 0..inf for mass_kg)."""
    _VALUE_RANGE[0], _VALUE_RANGE[1] = lo, hi


@dataclass
class QuantileForecast:
    quantiles: np.ndarray  # (B, V, H, Q)
    levels: tuple[float, ...]
    seconds: float = 0.0
    revision: str = "n/a"

    @property
    def median(self) -> np.ndarray:  # (B, V, H)
        return self.q(0.5)

    def q(self, level: float) -> np.ndarray:
        i = int(np.argmin(np.abs(np.asarray(self.levels) - level)))
        return self.quantiles[..., i]

    def has(self, level: float, tol: float = 1e-6) -> bool:
        return bool(np.min(np.abs(np.asarray(self.levels) - level)) < tol)

    def clipped(self) -> "QuantileForecast":
        return QuantileForecast(np.clip(self.quantiles, _VALUE_RANGE[0], _VALUE_RANGE[1]), self.levels, self.seconds, self.revision)


def _fill_gaps(ctx: np.ndarray) -> np.ndarray:
    """Linear interpolation + edge fill along the last axis, for models that reject NaN."""
    out = ctx.astype(np.float32).copy()
    flat = out.reshape(-1, out.shape[-1])
    x = np.arange(flat.shape[1])
    for row in flat:
        ok = np.isfinite(row)
        if ok.all():
            continue
        if not ok.any():
            row[:] = 0.0
            continue
        row[~ok] = np.interp(x[~ok], x[ok], row[ok])
    return flat.reshape(out.shape)


class Timestral:
    """Mistral Timestral (Modified-MIT weights). Accepts NaN gaps through the observation mask."""

    def __init__(self, repo: str = "mistralai/Timestral", device: str = "cpu", revision: str | None = None,
                 batch_size: int = 32, local_files_only: bool = False):
        from timestral import TimestralForecaster

        self.model = TimestralForecaster.from_pretrained(repo, device=device, revision=revision,
                                                         local_files_only=local_files_only)
        self.name = f"timestral:{repo.split('/')[-1]}"
        from importlib.metadata import version as _pkg_version
        self.revision = f"{repo}@{revision}" if revision else f"{repo} (timestral=={_pkg_version('timestral')})"
        self.levels = tuple(float(x) for x in self.model.quantile_levels)
        self.batch_size = batch_size
        self.clip = True

    def forecast(self, context: np.ndarray, horizon: int, known_covariates: np.ndarray | None = None,
                 past_covariates: np.ndarray | None = None) -> QuantileForecast:
        import torch

        t0 = time.perf_counter()
        outs = []
        for i in range(0, context.shape[0], self.batch_size):
            ctx = context[i : i + self.batch_size].astype(np.float32)
            cov = None
            if known_covariates is not None and self.model.supports_known_covariates:
                cov = known_covariates[i : i + self.batch_size].astype(np.float32)
            pcov = past_covariates[i : i + self.batch_size].astype(np.float32) if past_covariates is not None else None
            with torch.inference_mode():
                fc = self.model.forecast(
                    context=ctx, prediction_length=horizon, observation_mask=np.isfinite(ctx),
                    known_covariates=cov, past_covariates=pcov, symmetric_averaging=False, clip_nonnegative=True,
                )
            outs.append(fc.quantiles.float().cpu().numpy())
        q = np.concatenate(outs, axis=0)
        fc = QuantileForecast(q, self.levels, time.perf_counter() - t0, self.revision)
        return fc.clipped() if self.clip else fc


class TimesFM3:
    """Google TimesFM 3.0. Weights are non-commercial: benchmark use only."""

    def __init__(self, checkpoint: str = "google/timesfm-3.0-pytorch", device: str = "cpu", per_core_batch_size: int = 16):
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.model = TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=per_core_batch_size, device=device))
        self.name = "timesfm:3.0"
        self.revision = checkpoint
        self.levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        self.clip = True

    def forecast(self, context: np.ndarray, horizon: int, known_covariates: np.ndarray | None = None,
                 past_covariates: np.ndarray | None = None) -> QuantileForecast:
        t0 = time.perf_counter()
        contexts = [c for c in _fill_gaps(context)]  # each (V, T)
        covs = [c.astype(np.float32) for c in known_covariates] if known_covariates is not None else None
        pcovs = [c.astype(np.float32) for c in _fill_gaps(past_covariates)] if past_covariates is not None else None
        outs = list(self.model.predict_batch(contexts=contexts, horizon=horizon, past_future_covariates=covs,
                                             past_only_covariates=pcovs,
                                             return_quantiles=True, use_symmetric_averaging=False))
        q = np.stack([np.asarray(o.quantiles, dtype=np.float32) for o in outs], axis=0)
        if q.ndim == 3:  # (B, H, Q) for univariate -> (B, 1, H, Q)
            q = q[:, None]
        fc = QuantileForecast(q, self.levels, time.perf_counter() - t0, self.revision)
        return fc.clipped() if self.clip else fc


class Persistence:
    """Last observed value held flat. Point forecast only (single 0.5 level)."""

    name = "persistence"
    revision = "n/a"
    levels = (0.5,)

    def forecast(self, context: np.ndarray, horizon: int, known_covariates=None, past_covariates=None) -> QuantileForecast:
        filled = _fill_gaps(context)
        last = filled[..., -1]  # (B, V)
        q = np.repeat(last[..., None, None], horizon, axis=2)
        return QuantileForecast(q.astype(np.float32), self.levels).clipped()


class Ramp:
    """Linear fill-rate baseline: the context's mean hourly rate (lifts excluded) extrapolated
    from the last observed value and clipped to 100. A lift is a drop of more than
    ``reset_drop`` percentage points between consecutive observed buckets.
    """

    name = "ramp"
    revision = "n/a"
    levels = (0.5,)

    def __init__(self, reset_drop: float = 20.0, reset_frac: float | None = None):
        self.reset_drop = reset_drop
        self.reset_frac = reset_frac

    def _slope_and_last(self, row: np.ndarray) -> tuple[float, float]:
        ok = np.isfinite(row)
        if ok.sum() < 2:
            return 0.0, float(np.nan_to_num(row[-1]))
        x = np.flatnonzero(ok)
        y = row[ok]
        thr = self.reset_frac * np.percentile(y, 95) if self.reset_frac else self.reset_drop
        d = np.diff(y) / np.diff(x)
        d = d[d > -thr]
        slope = float(d.mean()) if len(d) else 0.0
        return max(slope, 0.0), float(y[-1])

    def forecast(self, context: np.ndarray, horizon: int, known_covariates=None, past_covariates=None) -> QuantileForecast:
        B, V, T = context.shape
        out = np.empty((B, V, horizon), dtype=np.float32)
        steps = np.arange(1, horizon + 1, dtype=np.float32)
        for b in range(B):
            for v in range(V):
                slope, last = self._slope_and_last(context[b, v].astype(np.float64))
                out[b, v] = last + slope * steps
        return QuantileForecast(out[..., None], self.levels).clipped()


def hour_of_week_covariates(index, horizon_index) -> np.ndarray:
    """Known-future covariates (C, T+H): sin/cos hour-of-day, sin/cos day-of-week, weekend flag."""
    import pandas as pd

    full = pd.DatetimeIndex(list(index) + list(horizon_index))
    h = full.hour.to_numpy() + full.minute.to_numpy() / 60.0
    d = full.dayofweek.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
        (d >= 5).astype(np.float32),
    ]).astype(np.float32)


class Cumulative:
    """Wrap a forecaster so it sees the fill series with lifts undone (a monotone cumulative
    inflow series) and forecasts inflow; the result is added to the last observed fill level.

    Drops larger than ``reset_drop`` between consecutive buckets are treated as lifts and
    removed; everything else (including sensor noise) is kept. The wrapped model therefore
    forecasts "how much comes in", and the question "when is it emptied" is left aside.
    """

    def __init__(self, inner, reset_drop: float = 20.0, reset_frac: float | None = None):
        self.inner = inner
        self.reset_frac = reset_frac  # if set, threshold = reset_frac * 95th percentile of each series
        inner.clip = False  # the cumulative series runs far above 100
        self.name = f"{inner.name}+cum"
        self.revision = inner.revision
        self.levels = inner.levels
        self.reset_drop = reset_drop

    def forecast(self, context: np.ndarray, horizon: int, known_covariates: np.ndarray | None = None,
                 past_covariates: np.ndarray | None = None) -> QuantileForecast:
        filled = _fill_gaps(context)
        d = np.diff(filled, axis=-1)
        if self.reset_frac:
            thr = self.reset_frac * np.percentile(filled, 95, axis=-1, keepdims=True)
        else:
            thr = self.reset_drop
        d = np.where(d < -thr, 0.0, d)
        cum = np.concatenate([filled[..., :1], filled[..., :1] + np.cumsum(d, axis=-1)], axis=-1).astype(np.float32)
        cum[~np.isfinite(context)] = np.nan  # keep the gap mask for models that use it
        fc = self.inner.forecast(cum, horizon, known_covariates, past_covariates)
        last_cum = _fill_gaps(cum)[..., -1]            # (B, V)
        last_fill = filled[..., -1]                    # (B, V)
        q = fc.quantiles - last_cum[..., None, None] + last_fill[..., None, None]
        q = np.maximum.accumulate(np.maximum(q, last_fill[..., None, None]), axis=2)  # inflow cannot be negative
        return QuantileForecast(q, fc.levels, fc.seconds, fc.revision).clipped()
