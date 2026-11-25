# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy"]
# ///
"""Illustrations for the probabilistic forecast slides.

Generates five figures in ``slides/assets/prob_forecast``:

- ``interval.png``: a synthetic daily load curve with its median forecast and
  the 90% prediction interval (5% and 95% quantiles). Roughly one observation
  in ten falls outside the interval, which is what a calibrated forecaster
  should produce. The curves are synthetic (shapes realistic, values not).
- ``pinball.png``: the pinball loss ``L_tau`` as a function of the error
  ``y - q`` for ``tau`` = 5%, 50% and 95%.
- ``coverage_vs_width_1h.png`` and ``coverage_vs_width_24h.png``: 90% coverage
  against interval width at the 1 h and 24 h horizons, one point per model.
- ``cost_vs_pinball_24h.png``: 24 h pinball loss against fit + predict time,
  with the Pareto frontier.

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


# 8 folds pooled. Tabular models: diff-feature runs (cv_comparison_forecast_fg3).
# Chronos-2 and t0-gamma: the comparison report (day-1 stitched runs), which do
# not use the lagged features.
# Each point is (name, width MW, coverage %, colour, marker).
COVERAGE_VS_WIDTH = {
    24: (
        ("HGB quantile", 3299, 76.23, BLUE, "o"),
        ("Chronos-2", 4373, 88.0, LIGHT_BLUE, "^"),
        ("HGB + recal", 4843, 91.05, ORANGE, "o"),
        ("HGB + conformal", 4868, 91.64, ORANGE, "s"),
        ("t0-gamma", 4974, 91.73, LIGHT_BLUE, "^"),
        ("TabICL", 5185, 87.47, BLUE, "D"),
        ("Kumo-Tabular", 6265, 98.45, BLUE, "D"),
    ),
    1: (
        ("HGB quantile", 1667, 85.42, BLUE, "o"),
        ("Chronos-2", 2203, 95.47, LIGHT_BLUE, "^"),
        ("HGB + recal", 1933, 91.05, ORANGE, "o"),
        ("HGB + conformal", 1933, 91.14, ORANGE, "s"),
        ("t0-gamma", 1253, 93.23, LIGHT_BLUE, "^"),
        ("TabICL", 1485, 91.51, BLUE, "D"),
        ("Kumo-Tabular", 1191, 91.03, BLUE, "D"),
    ),
}

# Label placement per horizon. ("off", dx, dy, ha): offset in points from the
# marker. ("at", x, y): text at data coordinates (coverage %, width MW), joined
# to the marker by a leader line; used for markers that sit too close together.
_LABELS = {
    24: {
        "HGB quantile": ("off", 8, 0, "left"),
        "Chronos-2": ("off", 0, -14, "center"),
        "TabICL": ("off", -8, 12, "right"),
        "Kumo-Tabular": ("off", -8, 10, "right"),
        "t0-gamma": ("at", 94.6, 5400),
        "HGB + conformal": ("at", 94.6, 4900),
        "HGB + recal": ("at", 94.6, 3900),
    },
    1: {
        "HGB quantile": ("off", 8, 0, "left"),
        "Chronos-2": ("off", 0, -14, "center"),
        "TabICL": ("off", 9, 0, "left"),
        "Kumo-Tabular": ("off", -4, -15, "left"),
        "t0-gamma": ("off", 9, 0, "left"),
        "HGB + conformal": ("at", 92.0, 2250),
        "HGB + recal": ("at", 87.2, 2250),
    },
}

_AXES = {
    # horizon: (xlim, ylim, nominal label (x, y, va))
    24: ((74, 101), (2900, 6900), (90.4, 3000, "bottom")),
    1: ((83, 97.5), (900, 2700), (90.2, 2680, "top")),
}


def plot_coverage_vs_width(path: Path, dpi: int, horizon: int) -> None:
    """Interval width against 90% coverage at one horizon, one point per model."""
    xlim, ylim, (nx, ny, nva) = _AXES[horizon]
    labels = _LABELS[horizon]
    fig, ax = plt.subplots(figsize=(9.2, 2.8), facecolor=BEIGE)
    _style(ax)
    ax.axvline(90, color=GRAY, lw=1, ls="--")
    ax.text(nx, ny, "nominal 90%", color=GRAY, fontsize=12, va=nva)
    for name, width, coverage, color, marker in COVERAGE_VS_WIDTH[horizon]:
        ax.scatter([coverage], [width], s=80, color=color, marker=marker, zorder=3, edgecolors=BLUE, linewidths=0.6)
        spec = labels[name]
        if spec[0] == "at":
            ax.annotate(
                name,
                (coverage, width),
                xytext=spec[1:],
                textcoords="data",
                ha="left",
                va="center",
                fontsize=12,
                color=BLUE,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkA=4, shrinkB=5),
            )
            continue
        _, dx, dy, ha = spec
        ax.annotate(
            name,
            (coverage, width),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
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
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"coverage at {horizon} h (%)", color=GRAY, fontsize=13)
    ax.set_ylabel("90% interval width (MW)", color=GRAY, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=BEIGE)
    plt.close(fig)


# 24 h horizon, 8 folds pooled. Cost: fit + predict time summed over the 8 folds,
# in seconds (diff-feature runs for the tabular models, the comparison report for
# the TSFMs). Pinball loss: mean of the 5% and 95% levels, in MW.
# Each point is (name, seconds, pinball MW, colour, marker).
COST_VS_PINBALL = (
    ("HGB quantile", 2062, 169.0, BLUE, "o"),
    ("HGB + recal", 9559, 156.8, ORANGE, "o"),
    ("HGB + conformal", 9520, 155.7, ORANGE, "s"),
    ("TabICL", 632, 171.0, BLUE, "D"),
    ("Kumo-Tabular", 1439, 160.7, BLUE, "D"),
    ("Chronos-2", 3106, 160.8, LIGHT_BLUE, "^"),
    ("t0-gamma", 1050, 174.2, LIGHT_BLUE, "^"),
)

# (dx, dy in points, ha): placed to stay off the frontier line
_COST_LABELS = {
    "HGB quantile": (9, 0, "left"),
    "HGB + recal": (9, 7, "left"),
    "HGB + conformal": (-9, -2, "right"),
    "TabICL": (0, 14, "center"),
    "Kumo-Tabular": (0, -15, "center"),
    "Chronos-2": (0, 14, "center"),
    "t0-gamma": (9, 0, "left"),
}


def pareto_frontier(points):
    """Non-dominated (cost, loss) pairs when both are minimised, by increasing cost."""
    frontier, best = [], float("inf")
    for cost, loss in sorted(points):
        if loss < best:
            frontier.append((cost, loss))
            best = loss
    return frontier


def plot_cost_vs_pinball(path: Path, dpi: int) -> None:
    """24 h pinball loss against total fit + predict time, with the Pareto frontier."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, ax = plt.subplots(figsize=(9.2, 3.1), facecolor=BEIGE)
    _style(ax)
    xmax = 300
    frontier = pareto_frontier([(sec / 60, loss) for _, sec, loss, _, _ in COST_VS_PINBALL])
    xs, ys = zip(*frontier)
    ax.step(
        [*xs, xmax], [*ys, ys[-1]], where="post", color=GRAY, lw=1.3, ls="--", zorder=1, label="Pareto frontier"
    )
    for name, sec, loss, color, marker in COST_VS_PINBALL:
        minutes = sec / 60
        ax.scatter([minutes], [loss], s=80, color=color, marker=marker, zorder=3, edgecolors=BLUE, linewidths=0.6)
        dx, dy, ha = _COST_LABELS[name]
        ax.annotate(
            name,
            (minutes, loss),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va="center",
            fontsize=12,
            color=BLUE,
        )
    # same colour key as the coverage plots, plus the frontier
    for label, color in (("raw tabular", BLUE), ("corrected HGB", ORANGE), ("TSFM", LIGHT_BLUE)):
        ax.scatter([], [], s=60, color=color, marker="o", edgecolors=BLUE, linewidths=0.6, label=label)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("raw tabular", "corrected HGB", "TSFM", "Pareto frontier")]
    legend = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        frameon=False,
        fontsize=11,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(GRAY)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(8, xmax)
    ax.set_ylim(153, 177.5)
    ax.set_xlabel("fit + predict time over the 8 folds (min, log scale)", color=GRAY, fontsize=13)
    ax.set_ylabel("24 h pinball loss (MW)", color=GRAY, fontsize=13)
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
    for horizon in (1, 24):
        plot_coverage_vs_width(args.outdir / f"coverage_vs_width_{horizon}h.png", args.dpi, horizon)
    plot_cost_vs_pinball(args.outdir / "cost_vs_pinball_24h.png", args.dpi)
    print(f"wrote figures in {args.outdir}")


if __name__ == "__main__":
    main()
