"""Presentation figure + table: how well does the forecast tell the fill level on lift day?

Reads outputs/showcase/per_event.csv (from covariate_test.py --leads 2,4,5,7) and writes
outputs/showcase/fill_at_lift_showcase.png and fill_at_lift_table.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fill_forecast.plot import SERIES_COLORS, INK, INK2, GRID

SURFACE = "#fcfcfb"
BEST = "timestral_cum@60+cal+mass+cycles"
LABELS = {BEST: "SMART forecast (Timestral + calendar + mass)", "timestral_cum": "Timestral, fill only",
          "timestral": "Timestral, raw series", "ramp": "linear fill-rate rule", "persistence": "no change (today's level)"}
LEAD_COLORS = {2: SERIES_COLORS[0], 4: SERIES_COLORS[1], 5: SERIES_COLORS[2], 7: SERIES_COLORS[3]}


def main(src="outputs/showcase", target="fill_pct"):
    global BEST
    src = Path(src)
    res = pd.read_csv(src / "per_event.csv")
    res["abs_err"] = res.fill_at_lift_err.abs()
    leads = sorted(res.lead_days.unique())
    is_mass = target == "mass_kg"
    if is_mass:
        BEST = "timestral_cum@60+cal+fill+cycles"
        LABELS[BEST] = "SMART forecast (Timestral + calendar + fill + cycles)"
        LABELS["ramp"] = "linear mass-rate rule"
        res["rel_err"] = res.abs_err / res.fill_at_lift_actual.clip(lower=1) * 100
    unit = "kg" if is_mass else "pp"
    best = res[res.experiment == BEST]

    # ---- table ----
    rows = []
    for lead in leads:
        r = best[best.lead_days == lead]
        if is_mass:
            wb = r.dropna(subset=["weighbridge_net_kg"])
            wb_rel = ((wb.fill_at_lift_pred - wb.weighbridge_net_kg) / wb.weighbridge_net_kg.clip(lower=1)).abs() * 100
            rows.append({
                "forecast issued": f"{lead} days before lift",
                "lifts": len(r),
                "typical error (MAE, kg)": round(r.abs_err.mean()),
                "half of forecasts within (%)": round(r.rel_err.median(), 1),
                "within 10 %": f"{100 * (r.rel_err <= 10).mean():.0f} %",
                "within 20 %": f"{100 * (r.rel_err <= 20).mean():.0f} %",
                "bias (kg)": round(r.fill_at_lift_err.mean()),
                "80 % band hit rate": f"{100 * r.covered_80.mean():.0f} %",
                "vs weighbridge ticket, half within (%)": round(wb_rel.median(), 1),
            })
            continue
        rows.append({
            "forecast issued": f"{lead} days before lift",
            "lifts": len(r),
            "typical error (MAE, pp)": round(r.abs_err.mean(), 1),
            "half of forecasts within (pp)": round(r.abs_err.median(), 1),
            "within 5 pp": f"{100 * (r.abs_err <= 5).mean():.0f} %",
            "within 10 pp": f"{100 * (r.abs_err <= 10).mean():.0f} %",
            "within 15 pp": f"{100 * (r.abs_err <= 15).mean():.0f} %",
            "bias (pp)": round(r.fill_at_lift_err.mean(), 1),
            "80 % band hit rate": f"{100 * r.covered_80.mean():.0f} %",
        })
    table = pd.DataFrame(rows)
    md = table.to_markdown(index=False)
    (src / ("mass_at_lift_table.md" if is_mass else "fill_at_lift_table.md")).write_text(md + "\n")
    print(md)

    # ---- figure ----
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28, left=0.05, right=0.985, top=0.88, bottom=0.08)

    for k, lead in enumerate(leads):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor(SURFACE)
        r = best[best.lead_days == lead]
        top = 8000 if is_mass else 100
        if is_mass:
            ax.fill_between([0, top], [0, top * 0.8], [0, top * 1.2], color=GRID, alpha=0.6, lw=0)
        else:
            ax.fill_between([0, 100], [-10, 90], [10, 110], color=GRID, alpha=0.6, lw=0)
        ax.plot([0, top], [0, top], color=INK2, lw=1)
        ax.scatter(r.fill_at_lift_actual, r.fill_at_lift_pred, s=16, color=LEAD_COLORS[lead], alpha=0.65, edgecolor="none")
        ax.set_title(f"{lead} days before the lift", loc="left", fontsize=12, color=INK, fontweight="bold")
        if is_mass:
            within20 = 100 * (r.rel_err <= 20).mean()
            ax.text(top * 0.03, top * 0.95, f"half of forecasts within {r.rel_err.median():.0f} %\n{within20:.0f} % of lifts within ±20 %",
                    fontsize=9.5, color=INK, va="top")
        else:
            within10 = 100 * (r.abs_err <= 10).mean()
            ax.text(3, 95, f"typical error ±{r.abs_err.mean():.0f} pp\n{within10:.0f} % of lifts within ±10 pp",
                    fontsize=9.5, color=INK, va="top")
        ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal")
        ax.set_xlabel(f"actual {'mass kg' if is_mass else 'fill %'} at the lift", color=INK2, fontsize=9.5)
        if k == 0:
            ax.set_ylabel(f"forecast {'mass kg' if is_mass else 'fill %'} at the lift", color=INK2, fontsize=9.5)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)

    # bottom left: error vs lead time, best vs references
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor(SURFACE)
    series = [(BEST, SERIES_COLORS[0], "-"), ("ramp", SERIES_COLORS[1], "--"), ("timestral", SERIES_COLORS[2], "--"), ("persistence", INK2, ":")]
    for name, c, ls in series:
        m = res[res.experiment == name].groupby("lead_days").abs_err.mean()
        ax.plot(m.index, m.values, color=c, lw=2.2 if name == BEST else 1.6, ls=ls, marker="o", ms=5, label=LABELS[name])
        ax.text(m.index[-1] + 0.15, m.values[-1], f"{m.values[-1]:.0f}", color=c, fontsize=9, va="center")
    ax.set_xticks(leads); ax.set_xticklabels([f"{l} d" for l in leads])
    ax.set_xlabel("forecast issued this many days before the lift", color=INK2, fontsize=9.5)
    ax.set_ylabel(f"mean error of {'mass at the lift (kg)' if is_mass else 'fill % at the lift (pp)'}", color=INK2, fontsize=9.5)
    ax.set_ylim(0, 1600 if is_mass else 40); ax.set_xlim(1.5, 7.8)
    ax.set_title("Error grows with lead time, and the right setup matters more than the model", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.tick_params(colors=INK2, labelsize=9)

    # bottom right: how often is the forecast within X points (ECDF per lead)
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor(SURFACE)
    xmax = 60 if is_mass else 30
    xs = np.linspace(0, xmax, 301)
    for lead in leads:
        e = (best[best.lead_days == lead].rel_err if is_mass else best[best.lead_days == lead].abs_err).to_numpy()
        ecdf = [(e <= x).mean() * 100 for x in xs]
        ax.plot(xs, ecdf, color=LEAD_COLORS[lead], lw=2, label=f"{lead} days before")
        ax.text(xmax * 1.01, ecdf[-1], f"{lead} d", color=LEAD_COLORS[lead], fontsize=9, va="center")
    for x in ((10, 20, 30) if is_mass else (5, 10, 15)):
        ax.axvline(x, color=GRID, lw=1)
    ax.set_xlim(0, xmax * 1.07); ax.set_ylim(0, 100)
    ax.set_xlabel("forecast error at the lift, % of actual mass" if is_mass else "forecast error at the lift, percentage points", color=INK2, fontsize=9.5)
    ax.set_ylabel("share of lifts with error at most this (%)", color=INK2, fontsize=9.5)
    ax.set_title("How often the forecast lands within 10 / 20 / 30 %" if is_mass else "How often the forecast lands within 5 / 10 / 15 points", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.tick_params(colors=INK2, labelsize=9)

    n = best.groupby("lead_days").size().max()
    if is_mass:
        fig.suptitle("How well do we know the mass on lift day in advance?", x=0.05, ha="left", fontsize=16, color=INK, fontweight="bold")
        fig.text(0.05, 0.915, f"{n} real collections on 23 compactors with a weight sensor, Jun–Sep 2026. Each dot is one lift: what the forecast said "
                 "the container mass would be at pickup vs what the scale read. Grey band = ±20 %.", fontsize=10.5, color=INK2)
        out = src / "mass_at_lift_showcase.png"
    else:
        fig.suptitle("How well do we know the fill level on lift day in advance?", x=0.05, ha="left", fontsize=16, color=INK, fontweight="bold")
        fig.text(0.05, 0.915, f"{n} real collections on 25 compactors, Jun–Sep 2026. Each dot is one lift: what the forecast said the fill % would be "
                 "at pickup vs what the sensor read. Grey band = ±10 points.", fontsize=10.5, color=INK2)
        out = src / "fill_at_lift_showcase.png"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
