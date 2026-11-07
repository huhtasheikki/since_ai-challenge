"""One PNG per device run: context, actuals, model medians with 10-90 bands, lifts, threshold."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import QuantileForecast

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]  # fixed order, never cycled
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def plot_run(
    path: str,
    device_id: str,
    context: pd.Series,
    actual: pd.Series,
    forecasts: dict[str, QuantileForecast],   # name -> quantiles indexed (H, Q) for this series
    metrics: pd.DataFrame,                    # rows with columns model, lead_bucket, mae
    events: pd.DataFrame | None = None,
    threshold: float | None = None,
    meta: dict | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(context.index, context.values, color=INK2, lw=1.2, label="observed (context, 30 d)")
    ax.plot(actual.index, actual.values, color=INK, lw=1.6, label="actual (holdout, 15 d)")

    mae_all = metrics[metrics["lead_bucket"] == "all"].set_index("model")["mae"] if not metrics.empty else {}
    for i, (name, fc) in enumerate(forecasts.items()):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        med = fc.q(0.5)
        label = f"{name}  MAE {mae_all.get(name, float('nan')):.1f}" if name in mae_all else name
        style = "-" if len(fc.levels) > 1 else "--"
        ax.plot(actual.index, med, color=c, lw=2, ls=style, label=label)
        if fc.has(0.1) and fc.has(0.9):
            ax.fill_between(actual.index, fc.q(0.1), fc.q(0.9), color=c, alpha=0.15, lw=0)

    cutoff = actual.index[0]
    ax.axvline(cutoff, color=INK2, lw=1, ls=":")
    ax.text(cutoff, 101, " cutoff", color=INK2, fontsize=9, va="bottom")
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color="#e34948", lw=1, ls=":", alpha=0.8)
        ax.text(context.index[0], threshold + 1, f"on-call threshold {threshold:.0f} %", color="#e34948", fontsize=9)
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            if ev["status"] in ("COMPLETED", "PARTIAL"):
                ax.axvline(ev["actual_at"], color="#008300", lw=0.8, alpha=0.5)

    title = f"{device_id} fill % forecast"
    if meta:
        bits = [str(meta.get("material_stream") or ""), str(meta.get("collection_policy") or ""),
                f"{meta.get('container_volume_m3')} m³" if meta.get("container_volume_m3") else ""]
        title += "  ·  " + ", ".join(b for b in bits if b)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel("fill_pct", color=INK2)
    ax.set_ylim(-2, 108)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
