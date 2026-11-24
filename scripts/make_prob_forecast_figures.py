# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy"]
# ///
"""Illustrations for the probabilistic forecast slides.

Generates two figures in ``slides/assets/prob_forecast``:

- ``interval.png``: a synthetic daily load curve with its median forecast and
  the 90% prediction interval (5% and 95% quantiles). Roughly one observation
  in ten falls outside the interval, which is what a calibrated forecaster
  should produce. The curves are synthetic (shapes realistic, values not).
- ``pinball.png``: the pinball loss ``L_tau`` as a function of the error
  ``y - q`` for ``tau`` = 5%, 50% and 95%.
- ``coverage_vs_width.png``: 90% coverage against interval width at the 24 h
  horizon, one point per model.

Usage::

    uv run scripts/make_prob_forecast_figures.py
    uv run scripts/make_prob_forecast_figures.py --outdir /tmp/figs --dpi 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# probabl.ai palette used by slides/style.css
BEIGE, BEIGE_DARK = "#f4f3ef", "#e4e1dd"
BLUE, ORANGE, LIGHT_BLUE = "#040524", "#ff7900", "#4cd0ff"
GRAY, RED = "#5b5b66", "#c0392b"

Z95 = 1.6449  # standard normal 95% quantile


def _style(ax):
    ax.set_facecolor(BEIGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=14)


def plot_interval(path: Path, dpi: int) -> None:
    """Median forecast and 90% prediction interval on a synthetic load curve."""
    rng = np.random.default_rng(13)
    hours = np.arange(72)
    daily = 55 - 7 * np.cos(2 * np.pi * (hours - 4) / 24) + 3 * np.sin(4 * np.pi * (hours - 8) / 24)
    weekly = -1.5 * (hours // 24)
    median = daily + weekly
    # the uncertainty is larger at the daily peaks than at night
    sigma = 1.0 + 0.8 * (1 + np.cos(2 * np.pi * (hours - 18) / 24)) / 2
    lo, hi = median - Z95 * sigma, median + Z95 * sigma
    truth = median + sigma * rng.standard_normal(hours.size)
    outside = (truth < lo) | (truth > hi)

    fig, ax = plt.subplots(figsize=(9.0, 3.3), facecolor=BEIGE)
    _style(ax)
    ax.fill_between(hours, lo, hi, color=LIGHT_BLUE, alpha=0.45, lw=0, label="90% prediction interval (5%-95%)")
    ax.plot(hours, lo, color=LIGHT_BLUE, lw=1)
    ax.plot(hours, hi, color=LIGHT_BLUE, lw=1)
    ax.plot(hours, median, color=ORANGE, lw=2.6, label="median forecast (50%)")
    ax.plot(hours, truth, color=BLUE, lw=1.0, alpha=0.55)
    ax.scatter(hours[~outside], truth[~outside], s=14, color=BLUE, zorder=3, label="observed load")
    ax.scatter(hours[outside], truth[outside], s=46, color=RED, zorder=4, label=f"outside the interval ({outside.sum()}/{hours.size})")
    ax.set_xlabel("time (hours)", color=GRAY, fontsize=13)
    ax.set_ylabel("load (GW)", color=GRAY, fontsize=13)
    ax.set_xlim(0, hours[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, fontsize=11.5, labelcolor=BLUE)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=BEIGE)
    plt.close(fig)


def pinball(u: np.ndarray, tau: float) -> np.ndarray:
    """Pinball loss ``max(tau * u, (tau - 1) * u)`` of the error ``u = y - q``."""
    return np.maximum(tau * u, (tau - 1) * u)


def plot_pinball(path: Path, dpi: int) -> None:
    """Pinball loss for tau = 5%, 50% and 95%."""
    u = np.linspace(-1, 1, 201)
    taus = (0.05, 0.5, 0.95)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), sharey=True, facecolor=BEIGE)
    for ax, tau in zip(axes, taus):
        _style(ax)
        ax.plot(u, pinball(u, tau), color=ORANGE, lw=2.8)
        ax.axvline(0, color=GRAY, lw=0.8, ls=":")
        ax.set_title(rf"$\tau$ = {tau:.0%}", color=BLUE, fontsize=17)
        ax.set_xlabel(r"error  $y - q$", color=GRAY, fontsize=14)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["y < q", "0", "y > q"])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(r"$L_\tau(y, q)$", color=GRAY, fontsize=13)
    axes[0].text(-0.45, 0.9, "q too high:\nsteep", fontsize=13.5, color=GRAY, va="top")
    axes[0].text(0.05, 0.42, "q too low:\nshallow", fontsize=13.5, color=GRAY, va="top")
    axes[1].text(0.0, 0.8, "symmetric:\nMAE / 2", fontsize=13.5, color=GRAY, va="top", ha="center")
    axes[2].text(-0.97, 0.42, "q too high:\nshallow", fontsize=13.5, color=GRAY, va="top")
    axes[2].text(-0.15, 0.9, "q too low:\nsteep", fontsize=13.5, color=GRAY, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=BEIGE)
    plt.close(fig)


# 24 h horizon, 8 folds pooled. Tabular models: diff-feature runs
# (cv_comparison_forecast_fg3). Chronos-2 and t0-gamma: the comparison report
# (day-1 stitched runs), which do not use the lagged features.
COVERAGE_VS_WIDTH = (
    ("HGB quantile", 3299, 76.23, BLUE, "o"),
    ("Chronos-2", 4373, 88.0, LIGHT_BLUE, "^"),
    ("HGB + recal", 4843, 91.05, ORANGE, "o"),
    ("HGB + conformal", 4868, 91.64, ORANGE, "s"),
    ("t0-gamma", 4974, 91.73, LIGHT_BLUE, "^"),
    ("TabICL", 5185, 87.47, BLUE, "D"),
    ("Kumo-Tabular", 6265, 98.45, BLUE, "D"),
)

# label offsets in points. Coverage is now horizontal, so the 91% cluster
# stacks vertically and the labels sit beside it.
_LABEL_OFFSET = {
    "HGB quantile": (8, 0),
    "Chronos-2": (0, -14),
    "TabICL": (-8, 12),
    "Kumo-Tabular": (-8, 10),
}
# the 91% cluster gets labels in free space on the right, joined by leader lines
# (data coordinates: coverage %, width MW), stacked in the same order as the points
_LABEL_AT = {
    "t0-gamma": (94.6, 5400),
    "HGB + conformal": (94.6, 4900),
    "HGB + recal": (94.6, 3900),
}
_LABEL_HA = {
    "HGB quantile": "left",
    "Chronos-2": "center",
    "TabICL": "right",
    "Kumo-Tabular": "right",
}


def plot_coverage_vs_width(path: Path, dpi: int) -> None:
    """Interval width against 90% coverage at 24 h, one point per model."""
    fig, ax = plt.subplots(figsize=(9.2, 2.8), facecolor=BEIGE)
    _style(ax)
    ax.axvline(90, color=GRAY, lw=1, ls="--")
    ax.text(90.4, 3000, "nominal 90%", color=GRAY, fontsize=12, va="bottom")
    for name, width, coverage, color, marker in COVERAGE_VS_WIDTH:
        ax.scatter([coverage], [width], s=80, color=color, marker=marker, zorder=3, edgecolors=BLUE, linewidths=0.6)
        if name in _LABEL_AT:
            ax.annotate(
                name,
                (coverage, width),
                xytext=_LABEL_AT[name],
                textcoords="data",
                ha="left",
                va="center",
                fontsize=12,
                color=BLUE,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkA=4, shrinkB=5),
            )
            continue
        dx, dy = _LABEL_OFFSET[name]
        ax.annotate(
            name,
            (coverage, width),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=_LABEL_HA[name],
            va="center",
            fontsize=12,
            color=BLUE,
        )
    # colour key: tabular models are navy, corrected HGB orange, TSFMs light blue
    for label, color in (("raw tabular", BLUE), ("corrected HGB", ORANGE), ("TSFM", LIGHT_BLUE)):
        ax.scatter([], [], s=60, color=color, marker="o", edgecolors=BLUE, linewidths=0.6, label=label)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=11, handletextpad=0.3, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(GRAY)
    ax.set_xlim(74, 101)
    ax.set_ylim(2900, 6900)
    ax.set_xlabel("coverage at 24 h (%)", color=GRAY, fontsize=13)
    ax.set_ylabel("90% interval width (MW)", color=GRAY, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=BEIGE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "slides/assets/prob_forecast")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_interval(args.outdir / "interval.png", args.dpi)
    plot_pinball(args.outdir / "pinball.png", args.dpi)
    plot_coverage_vs_width(args.outdir / "coverage_vs_width.png", args.dpi)
    print(f"wrote figures in {args.outdir}")


if __name__ == "__main__":
    main()
