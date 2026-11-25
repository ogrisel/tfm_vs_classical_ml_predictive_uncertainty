# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy"]
# ///
"""Step-by-step diagram: fitting 5 quantiles by minimizing the pinball loss.

Generates one image per animation frame (10 frames). Each frame shares the same
layout as the previous one and only adds or moves a few elements, so that going
from one slide to the next looks like an animation (see the generated
``slides_snippet.md``).

The diagram explains how to estimate the quantiles ``tau`` = 5, 25, 50, 75 and
95% of a log-normal sample *without sorting it*, starting from the prediction
``q = 0`` and repeatedly applying the same incremental update (a full-batch
subgradient descent step on the mean pinball loss)::

    q <- q - eta * (share of the sample below q - tau)

This is what a quantile regression model does behind the scenes, with ``q`` the
output of the model instead of a free parameter. The sample is synthetic.

Usage::

    uv run scripts/make_pinball_descent_frames.py            # PNG frames in slides/assets/pinball_descent
    uv run scripts/make_pinball_descent_frames.py --format svg
    uv run scripts/make_pinball_descent_frames.py --frames 4 5 --outdir /tmp/frames
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Style: probabl.ai palette used by slides/style.css
# --------------------------------------------------------------------------
BEIGE, BEIGE_DARK = "#f4f3ef", "#e4e1dd"
BLUE, ORANGE, LIGHT_BLUE = "#040524", "#ff7900", "#4cd0ff"
GRAY = "#5b5b66"
# one colour per quantile level, reused in every panel
TAU_COLORS = ["#1a8fc4", "#7a4fb5", ORANGE, "#d6336c", "#2e8b57"]


def _fonts(preferred, fallback):
    available = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in preferred if f in available] + [fallback]


SERIF = _fonts(["IBM Plex Serif", "Georgia"], "DejaVu Serif")
SANS = _fonts(["Inter", "Helvetica Neue", "Arial"], "DejaVu Sans")
MONO = _fonts(["IBM Plex Mono", "Menlo", "Courier New"], "DejaVu Sans Mono")
plt.rcParams.update(
    {"font.family": SANS, "svg.fonttype": "path", "axes.unicode_minus": True}
)

# Canvas in inches. 16:9 like the slides (remark ratio "16:9"), sized so that
# 12 pt text stays readable once the frame is scaled to the slide.
W, HT = 12.8, 7.2
LEFT, WIDTH = 0.45, 11.9

# --------------------------------------------------------------------------
# The problem and the optimization path
# --------------------------------------------------------------------------
SIGMA = 0.75  # y = exp(SIGMA * z), z ~ N(0, 1): median 1, mean 1.32
N_SAMPLES = 200
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
ETA = 1.0  # step size, in the units of y
N_ITER = 50
Q_MAX = 4.0  # right end of the loss curves
X_MAX = 6.0  # right end of the histogram


def pinball(y, q, tau):
    """Pinball loss ``max(tau * (y - q), (tau - 1) * (y - q))``."""
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


@dataclass
class Problem:
    y: np.ndarray
    taus: np.ndarray
    path: np.ndarray  # (N_ITER + 1, n_taus): q after each iteration, row 0 is q = 0
    exact: np.ndarray  # the minimizers of the mean pinball loss (sample quantiles)
    qs: np.ndarray  # grid for the loss curves
    curves: np.ndarray  # (n_taus, len(qs)): mean pinball loss on the grid

    def share_below(self, q):
        """Fraction of the sample strictly below each prediction (empirical CDF)."""
        return (self.y[:, None] < np.asarray(q)).mean(axis=0)

    def mean_loss_at(self, qs, i):
        """Mean pinball loss of level ``i`` at each of the points ``qs``."""
        return pinball(self.y[:, None], np.asarray(qs)[None, :], self.taus[i]).mean(axis=0)

    def mean_loss(self, q):
        """Mean pinball loss of the predictions ``q`` (one value per level)."""
        return np.array(
            [pinball(self.y, q[i], t).mean() for i, t in enumerate(self.taus)]
        )


def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.lognormal(mean=0.0, sigma=SIGMA, size=N_SAMPLES)
    taus = np.array(TAUS)
    path = np.zeros((N_ITER + 1, len(taus)))
    for k in range(N_ITER):
        q = path[k]
        # subgradient of the mean pinball loss: (share below q) - tau
        grad = (y[:, None] < q).mean(axis=0) - taus
        path[k + 1] = q - ETA * grad
    qs = np.linspace(-0.1, Q_MAX, 411)
    curves = np.array(
        [pinball(y[:, None], qs[None, :], t).mean(axis=0) for t in taus]
    )
    # the minimizer of the mean pinball loss is the sample quantile
    exact = np.array([np.quantile(y, t, method="inverted_cdf") for t in taus])
    return Problem(y, taus, path, exact, qs, curves)


# --------------------------------------------------------------------------
# Frame definitions: title, subtitle, speaker notes, iteration shown (or None)
# --------------------------------------------------------------------------
FRAMES = [
    (
        "A skewed sample, 5 quantiles to estimate",
        "200 draws of a log-normal variable y: asymmetric, with a long right tail.",
        "Synthetic data: y = exp(0.75 z) with z standard normal, so the median is about 1 and "
        "the mean about 1.3. We want 5 numbers q such that a fraction tau = 5, 25, 50, 75 and 95% of the "
        "sample lies below q. Sorting the sample would do for a fixed sample, but a quantile "
        "regression model has to learn q(x) with gradients: we need a loss whose minimizer "
        "is the quantile.",
        None,
    ),
    (
        "The pinball loss of one observation",
        "Penalty for predicting q when y = 1 is observed: a tilted absolute value.",
        "A single observation y = 1. Predicting too low costs tau per unit of error, predicting "
        "too high costs 1 - tau. For tau = 5% overshooting is 19 times more expensive than "
        "undershooting, which pushes q down; for 95% it is the opposite. At tau = 50% both "
        "sides cost the same: this is half the absolute error.",
        None,
    ),
    (
        "Averaged over the sample: convex, minimum at the quantile",
        "The mean pinball loss is piecewise linear in q, one kink per observation.",
        "Averaging over the 200 observations gives one convex curve per tau. Its slope at q is "
        "(share of the sample below q) - tau: it vanishes exactly when the share below q equals "
        "tau, i.e. at the sample tau-quantile. So the quantile is the minimizer. We could scan "
        "the curve; instead we only use the slope at the current q, which is what gradient "
        "descent (and the training of a model) has access to.",
        None,
    ),
    (
        "Why the curve looks smooth: zoom on the slope",
        "The slope is a staircase with one step of 1/n per observation; the loss integrates it.",
        "Zoom on the median (50%) curve, on a window that contains 12 observations. Top row: the "
        "mean loss, bottom row: its slope, which is the share of the sample below q minus tau, "
        "i.e. the empirical CDF shifted by tau. Each observation makes the slope jump by 1/n = "
        "0.005, so the loss has a kink there, but a kink that small is invisible at full scale: "
        "the staircase looks like a smooth curve and its integral, the loss, looks smooth too. "
        "Also note that the slope crosses zero exactly where the share below q reaches tau.",
        None,
    ),
    (
        "Start from the prediction q = 0",
        "All five predictions begin at 0, below every observation.",
        "Initialization: q = 0 for the 5 levels, as for a model that outputs zeros. Every "
        "observation is above q, so the share below is 0 and the slope is 0 - tau = -tau: "
        "the steeper for the high quantiles. The tangent drawn on each curve shows that "
        "increasing q reduces the loss.",
        0,
    ),
    (
        "Iteration 1: step by tau",
        "Incremental update q ← q − η · (share below − τ), with η = 1.",
        "First update: nothing is below q, so every quantile moves up by eta * tau: 0.05, "
        "0.25, 0.50, 0.75, 0.95. High quantiles take the biggest steps because they are "
        "the furthest from 0.",
        1,
    ),
    (
        "Iteration 2: the steps start to differ",
        "The share below q is now positive for the four upper levels: smaller steps.",
        "Some observations are now below the 25 to 95% predictions, so the gap between the "
        "share below and tau, hence the step, shrinks. The 5% quantile still has nothing "
        "below it: it takes the same tiny step of 0.05 again. The steps are never larger "
        "than tau: the update is bounded whatever the scale of y.",
        2,
    ),
    (
        "Iteration 3: the middle quantiles settle",
        "Each prediction slows down as the share below q gets close to τ.",
        "The 25 and 50% quantiles are within about 10% of their final value and the tangent "
        "on their curves is getting flat; the 75 and 95% quantiles are still travelling. The "
        "updates are driven by the gap between the share below q and tau: no sorting, no "
        "assumption on the distribution.",
        3,
    ),
    (
        "Iteration 10: the tails are still moving",
        "Where the sample is sparse the slope is small, so is the step.",
        "After 10 iterations the 25, 50 and 75% quantiles are done. The 5 and 95% tails keep "
        "moving: few observations sit around them, so q has to travel across a low density "
        "region and the share below changes slowly. Better optimizers (momentum, Adam, "
        "line search) or a step size matched to the data scale speed this up.",
        10,
    ),
    (
        "Iteration 50: five quantiles, a 90% interval",
        "The share below each q matches τ, the slope is ≈ 0: the loss is at its minimum.",
        "After 50 iterations, the predictions match the minimizers of the mean pinball loss "
        "(the sample quantiles, hollow markers), the upper tail being the slowest to converge. "
        "The 5% and 95% quantiles delimit a 90% prediction interval. A quantile regression "
        "model does the same, with q = f(x; theta) and the gradient flowing into theta.",
        N_ITER,
    ),
]
N_FRAMES = len(FRAMES)


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def new_canvas(dpi, title, subtitle):
    fig = plt.figure(figsize=(W, HT), dpi=dpi, facecolor=BEIGE)
    bg = fig.add_axes([0, 0, 1, 1], zorder=20)  # overlay in inches, y pointing up
    bg.set_xlim(0, W)
    bg.set_ylim(0, HT)
    bg.axis("off")
    bg.patch.set_alpha(0)
    bg.text(LEFT, 6.95, title, family=SERIF, size=26, color=BLUE, va="top")
    bg.add_patch(Rectangle((LEFT, 6.42), 0.6, 0.05, color=ORANGE, lw=0))
    bg.text(LEFT, 6.3, subtitle, size=14, color=GRAY, va="top")
    return fig, bg


def axes_in(fig, left, bottom, width, height, **kw):
    return fig.add_axes([left / W, bottom / HT, width / W, height / HT], **kw)


def box(bg, x, y, w, h, fc, ec=None, lw=1.2, alpha=1.0, radius=0.06, z=1):
    bg.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=fc,
            ec=ec if ec else "none",
            lw=lw,
            alpha=alpha,
            zorder=z,
        )
    )


def caption(bg, lines, y0=0.15):
    """White box with monospace lines, from ``y0`` up."""
    h = 0.22 + 0.33 * len(lines)
    box(bg, LEFT, y0, WIDTH, h, "white", BEIGE_DARK, lw=1.5, alpha=0.8)
    for i, line in enumerate(lines):
        bg.text(
            LEFT + 0.2, y0 + h - 0.13 - 0.33 * i, line, family=MONO, size=12.5,
            color=BLUE, va="top",
        )


def style_axes(ax):
    ax.patch.set_alpha(0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.tick_params(axis="x", length=3, colors=GRAY, labelsize=11)


def tau_label(tau):
    return f"{tau:.0%}"


# --------------------------------------------------------------------------
# Top panel: the sample and the five predictions
# --------------------------------------------------------------------------
BINS = np.arange(0, X_MAX + 1e-9, 0.2)
RAIL_PITCH = 0.22  # inches between the rows of predictions


def draw_data(fig, P, bottom, height, *, rails=False, k=None, targets=False, interval=False):
    """Histogram of the sample; with ``rails``, one row per level below it.

    A row shows where the prediction of that level currently is (big dot), the
    previous iterates (small dots) and, for the first iterations, the last step
    as an arrow. ``targets`` adds the minimizers of the loss as hollow markers.
    """
    ax = axes_in(fig, LEFT + 0.3, bottom, WIDTH - 0.6, height)
    style_axes(ax)
    n = len(P.y)
    counts, _ = np.histogram(P.y, bins=BINS)
    dens = counts / (n * 0.2)  # density, the bars integrate to the share shown
    hmax = dens.max()
    n_rows = len(P.taus) if rails else 0
    rails_in = n_rows * RAIL_PITCH + 0.25 if rails else 0.3
    per_inch = hmax * 1.15 / (height - rails_in)
    ymin, ymax = -rails_in * per_inch, hmax * 1.15
    ax.set_xlim(-0.12, X_MAX)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(int(X_MAX) + 1))
    ax.tick_params(axis="x", length=0, pad=4)
    ax.spines["bottom"].set_visible(False)

    ax.bar(BINS[:-1], dens, width=0.2, align="edge", color=LIGHT_BLUE, alpha=0.5,
           ec=BEIGE, lw=0.8, zorder=1)
    ax.axhline(0, color=GRAY, lw=1, zorder=2)
    # rug: one tick per observation, just below the axis
    tick = 0.12 * per_inch
    ax.vlines(P.y[P.y < X_MAX], -tick, 0, color=BLUE, alpha=0.4, lw=0.9, zorder=2)
    ax.text(X_MAX, hmax * 1.12, "y", ha="right", va="top", size=13, color=GRAY, style="italic")
    n_out = int((P.y >= X_MAX).sum())
    if n_out:
        ax.text(X_MAX, hmax * 0.12, f"+{n_out} beyond", ha="right", size=10.5, color=GRAY)

    if not rails:
        # frame 1: mean and median show the skew
        for val, name, dy in ((np.median(P.y), "median", 0.0), (P.y.mean(), "mean", 0.16)):
            ax.axvline(val, color=GRAY, lw=1.4, ls="--", zorder=3)
            ax.text(val + 0.05, hmax * (1.08 - dy), f"{name} {val:.2f}", size=12, color=GRAY,
                    va="top", bbox=dict(fc=BEIGE, ec="none", alpha=0.85, pad=1.5))
        return ax

    rows = [-(0.2 + RAIL_PITCH * i) * per_inch for i in range(n_rows)]
    q_now = P.path[k] if k is not None else None
    if interval and q_now is not None:
        ax.axvspan(q_now[0], q_now[-1], ymin=0, ymax=1, color=ORANGE, alpha=0.07, lw=0, zorder=0)
        ax.text((q_now[0] + q_now[-1]) / 2 + 0.3, hmax * 1.08,
                "90% of the observations", ha="center", va="top", size=12, color=ORANGE,
                weight="bold")
    for i, (tau, y_row) in enumerate(zip(P.taus, rows)):
        c = TAU_COLORS[i]
        ax.axhline(y_row, color=c, lw=0.8, alpha=0.18, zorder=1)
        ax.text(-0.18, y_row, tau_label(tau), ha="right", va="center", size=11.5, color=c,
                weight="bold", clip_on=False)
        if targets:
            ax.plot([P.exact[i]], [y_row], "D", ms=11, mfc="none", mec=c, mew=1.8, zorder=4)
        if q_now is None:
            continue
        ax.plot([q_now[i]] * 2, [y_row, hmax * 1.02], color=c, lw=1.4, alpha=0.7, zorder=3)
        ax.plot(P.path[: k + 1, i], [y_row] * (k + 1), "o", ms=3.5, color=c, alpha=0.45, zorder=4)
        if k and k <= 3:
            ax.annotate("", xy=(P.path[k, i], y_row), xytext=(P.path[k - 1, i], y_row),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4, shrinkA=0, shrinkB=0),
                        zorder=5)
        ax.plot([q_now[i]], [y_row], "o", ms=10, color=c, mec="white", mew=1.2, zorder=6)
    return ax


# --------------------------------------------------------------------------
# Bottom panels: one loss curve per level
# --------------------------------------------------------------------------
PANEL_GAP = 0.25
PANEL_W = (WIDTH - 4 * PANEL_GAP) / 5


def panel_axes(fig, i, bottom, height):
    left = LEFT + i * (PANEL_W + PANEL_GAP)
    ax = axes_in(fig, left, bottom, PANEL_W, height)
    style_axes(ax)
    ax.set_xticks(range(int(Q_MAX) + 1))
    return ax, left


def panel_header(bg, left, i, tau, line2, bottom, height):
    """Level (bold, coloured) and one line of numbers above the panel."""
    c = TAU_COLORS[i]
    y = bottom + height
    bg.text(left, y + 0.34, f"τ = {tau_label(tau)}", size=14, color=c, weight="bold", va="bottom")
    if line2:
        bg.text(left, y + 0.1, line2, size=11, color=BLUE, va="bottom", family=MONO)


def draw_kernel_panels(fig, bg, bottom, height):
    """Pinball loss of a single observation y = 1, as a function of q."""
    qs = np.linspace(0, 2, 201)
    for i, tau in enumerate(TAUS):
        c = TAU_COLORS[i]
        ax, left = panel_axes(fig, i, bottom, height)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0", "y = 1", "2"])
        ax.plot(qs, pinball(1.0, qs, tau), color=c, lw=3)
        ax.axvline(1, color=GRAY, lw=0.9, ls=":")
        panel_header(bg, left, i, tau, f"slope {-tau:+.2f} | {1 - tau:+.2f}", bottom, height)
        ax.text(0.5, 1.0, "q < y", ha="center", size=11, color=GRAY, va="top")
        ax.text(1.5, 1.0, "q > y", ha="center", size=11, color=GRAY, va="top")
    ax0 = fig.axes[-5]
    ax0.set_ylabel("loss", size=12, color=GRAY)


def draw_loss_panels(fig, bg, P, bottom, height, *, k=None, minima=False, numbers=True):
    """Mean pinball loss as a function of q, with the iterates and the tangent."""
    for i, tau in enumerate(P.taus):
        c = TAU_COLORS[i]
        ax, left = panel_axes(fig, i, bottom, height)
        curve = P.curves[i]
        ax.set_xlim(P.qs[0], Q_MAX)
        ax.set_ylim(0, curve[0] * 1.12 if curve[0] > curve.min() else curve.max() * 1.1)
        ax.plot(P.qs, curve, color=c, lw=2.6, zorder=2)
        line2 = ""
        if minima:
            ax.plot([P.exact[i]], [pinball(P.y, P.exact[i], tau).mean()], "D", ms=10,
                    mfc="none", mec=c, mew=1.8, zorder=4)
        if k is not None:
            q = P.path[k]
            share = P.share_below(q)
            lq = P.mean_loss(q)
            slope = share[i] - tau
            ax.plot(P.path[: k + 1, i], [np.interp(v, P.qs, curve) for v in P.path[: k + 1, i]],
                    "o", ms=4, color=c, alpha=0.4, zorder=3)
            # tangent: slope = share below - tau
            x0 = q[i]
            half = 0.7
            xs = np.array([x0 - half, x0 + half])
            ax.plot(xs, lq[i] + slope * (xs - x0), color=BLUE, lw=1.6, ls="-", zorder=4,
                    alpha=0.8, clip_on=True)
            ax.plot([x0], [lq[i]], "o", ms=10, color=c, mec="white", mew=1.2, zorder=6)
            line2 = f"q={x0:4.2f} {share[i]:4.0%}<q" if numbers else ""
        panel_header(bg, left, i, tau, line2, bottom, height)
    fig.axes[-5].set_ylabel("mean loss", size=11, color=GRAY)



# --------------------------------------------------------------------------
# Zoom on the kinks of one loss curve and on the staircase slope
# --------------------------------------------------------------------------
def staircase(P, lo, hi, tau):
    """Exact slope of the mean loss on ``[lo, hi]``: (share below q) - tau, as steps."""
    ys = np.sort(P.y)
    obs = ys[(ys > lo) & (ys < hi)]
    base = int((ys <= lo).sum())
    xs = np.concatenate([[lo], obs, [hi]])
    vals = (base + np.arange(len(obs) + 1)) / len(ys) - tau
    return obs, xs, np.append(vals, vals[-1])


def draw_zoom(fig, bg, P, i=2, n_zoom=12):
    """Loss (top row) and slope (bottom row): full range (left), zoom (right)."""
    tau, c = P.taus[i], TAU_COLORS[i]
    n = len(P.y)
    ys = np.sort(P.y)
    j = int(np.searchsorted(ys, P.exact[i]))
    a, b = j - n_zoom // 2, j + n_zoom // 2
    lo, hi = (ys[a - 1] + ys[a]) / 2, (ys[b - 1] + ys[b]) / 2
    rows = {"loss": (3.55, 1.9), "slope": (1.55, 1.5)}
    cols = {"full": (LEFT + 0.55, 4.9), "zoom": (LEFT + 6.55, 5.3)}
    axs = {}
    for r, (bottom, height) in rows.items():
        for cname, (left, width) in cols.items():
            ax = axes_in(fig, left, bottom, width, height)
            style_axes(ax)
            axs[r, cname] = ax
    # --- full range
    full = (P.qs[0], Q_MAX)
    obs, xs, vals = staircase(P, 0.0, Q_MAX, tau)
    axs["loss", "full"].plot(P.qs, P.curves[i], color=c, lw=2.6)
    axs["slope", "full"].step(xs, vals, where="post", color=c, lw=2.2)
    for r in ("loss", "slope"):
        ax = axs[r, "full"]
        ax.set_xlim(*full)
        ax.axvspan(lo, hi, color=ORANGE, alpha=0.25, lw=0)
    axs["loss", "full"].set_ylim(0, P.curves[i][0] * 1.1)
    axs["loss", "full"].set_xticklabels([])
    axs["slope", "full"].set_ylim(-0.58, 0.55)
    axs["slope", "full"].axhline(0, color=GRAY, lw=1, ls=":")
    axs["slope", "full"].set_xticks(range(int(Q_MAX) + 1))
    # --- zoom
    pts = np.concatenate([[lo], ys[(ys > lo) & (ys < hi)], [hi]])
    loss = P.mean_loss_at(pts, i)
    obs, xs, vals = staircase(P, lo, hi, tau)
    ax = axs["loss", "zoom"]
    ax.plot(pts, loss, color=c, lw=2.6, marker="o", ms=5, mec="white", mew=0.8)
    pad = 0.12 * (loss.max() - loss.min())
    ax.set_ylim(loss.min() - pad, loss.max() + pad)
    ax.set_xlim(lo, hi)
    ax.set_xticklabels([])
    ax = axs["slope", "zoom"]
    ax.step(xs, vals, where="post", color=c, lw=2.6)
    ax.axhline(0, color=GRAY, lw=1, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(vals.min() - 0.012, vals.max() + 0.012)
    ax.set_xticks(np.round(np.linspace(lo, hi, 4), 2))
    for r in ("loss", "slope"):
        zax = axs[r, "zoom"]
        zax.vlines(obs, 0, 1, transform=zax.get_xaxis_transform(), color=BLUE,
                   alpha=0.25, lw=0.8, ls=":", zorder=0)
    # one step of the staircase: +1/n
    m = len(obs) // 2
    x = obs[m]
    v1 = vals[m + 1]
    xb = x + 0.003 * (hi - lo) / 0.12
    ax.plot([xb, xb], [v1 - 1 / n, v1], color=BLUE, lw=3, solid_capstyle="butt", zorder=5)
    ax.text(xb + 0.002, v1 - 0.5 / n - 0.012, f"step 1/n = {1 / n:.3f}", size=12, color=BLUE,
            va="top", bbox=dict(fc=BEIGE, ec="none", alpha=0.9, pad=1))
    # titles and labels
    bg.text(cols["full"][0], 5.72, f"τ = {tau_label(tau)}, full range", size=14, color=c,
            weight="bold", va="bottom")
    bg.text(cols["zoom"][0], 5.72, f"zoom: the {len(obs)} observations in the shaded window",
            size=14, color=ORANGE, weight="bold", va="bottom")
    for r, (bottom, height) in rows.items():
        label = "mean loss" if r == "loss" else "slope of the loss"
        bg.text(LEFT + 0.3, bottom + height / 2, label, size=11.5, color=GRAY, rotation=90,
                va="center", ha="center")
    bg.text(cols["zoom"][0] + cols["zoom"][1] - 0.1, 5.4, "one kink per observation", size=12,
            color=c, style="italic", va="top", ha="right")
    bg.text(cols["zoom"][0] + 0.1, 3.0, "one step per observation", size=12, color=c,
            style="italic", va="top")

# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def draw_frame(step, P, dpi):
    title, subtitle, _, k = FRAMES[step - 1]
    fig, bg = new_canvas(dpi, title, subtitle)

    if step == 1:
        draw_data(fig, P, 1.8, 4.1)
        caption(bg, [
            f"y_i ~ LogNormal(0, {SIGMA})   i = 1 ... {N_SAMPLES}     (mean > median: long right tail)",
            "goal: q_τ for τ = 5, 25, 50, 75, 95 %   with   share of the sample below q_τ ≈ τ",
        ], y0=0.5)
    elif step == 2:
        draw_kernel_panels(fig, bg, 2.2, 2.8)
        caption(bg, [
            "L_τ(y, q) = τ · (y − q)          if y ≥ q     q too low:  slope −τ",
            "          = (1 − τ) · (q − y)    if y <  q     q too high: slope 1 − τ",
            "τ = 50 %: both sides cost the same, L = |y − q| / 2",
        ], y0=0.45)
    elif step == 3:
        draw_data(fig, P, 3.6, 2.45, rails=True, targets=True)
        draw_loss_panels(fig, bg, P, 1.5, 1.25, minima=True)
        caption(bg, [
            "mean loss(q) = 1/n Σ L_τ(y_i, q)     slope at q = (share of y below q) − τ",
            "slope = 0 ⇔ share below q = τ  ⇔  q is the sample τ-quantile (◇)",
        ], y0=0.15)
    elif step == 4:
        draw_zoom(fig, bg, P)
        caption(bg, [
            "slope(q) = share of y below q − τ     one step of +1/n at each observation",
            "loss(q) = integral of the slope       one kink of size 1/n per observation: invisible",
        ], y0=0.15)
    else:
        draw_data(fig, P, 3.6, 2.45, rails=True, k=k, targets=(step == N_FRAMES),
                  interval=(step == N_FRAMES))
        draw_loss_panels(fig, bg, P, 1.5, 1.25, k=k)
        steps = P.path[k] - P.path[k - 1] if k else None
        if k == 0:
            lines = [
                "q = 0 for every τ  →  every observation is above q: share below = 0",
                "slope = share below − τ = −τ  <  0: the loss decreases when q increases",
            ]
        elif step == N_FRAMES:
            lines = [
                f"after {k} iterations: share below q = "
                + "  ".join(f"{s:.0%}" for s in P.share_below(P.path[k]))
                + "  (τ = 5 25 50 75 95 %)",
                "q = " + " ".join(f"{v:.2f}" for v in P.path[k])
                + "     ◇ minimizers: " + " ".join(f"{v:.2f}" for v in P.exact),
            ]
        else:
            lines = [
                f"q ← q − η · (share below q − τ)      η = {ETA:g}"
                ,
                "step = " + "  ".join(f"{v:+.2f}" for v in steps) + "   (τ = 5 25 50 75 95 %)",
            ]
        caption(bg, lines, y0=0.15)
        if k and step != N_FRAMES:
            bg.text(W - LEFT, 6.95, f"iteration {k}", size=15, color=ORANGE, weight="bold",
                    ha="right", va="top", family=MONO)
    return fig


def slides_snippet(rel, fmt):
    """remark.js markdown: one slide per frame, the frame is the slide background."""
    # Generated by scripts/make_pinball_descent_frames.py: paste into slides/index.html.
    out = []
    for i, (title, _, notes, _) in enumerate(FRAMES, start=1):
        out += [
            "---",
            f"background-image: url({rel}/frame_{i:02d}.{fmt})",
            # style.css positions the logo with an offset: reset it for full-bleed frames
            "background-position: center",
            "background-repeat: no-repeat",
            "background-size: 100% 100%",
            "",
            "???",
            "",
            f"- {title}. {notes}",
            "",
        ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", type=Path, default=REPO / "slides/assets/pinball_descent")
    ap.add_argument("--format", choices=["png", "svg"], default="png")
    ap.add_argument("--dpi", type=int, default=150, help="150 dpi gives 1920x1080 px")
    ap.add_argument("--seed", type=int, default=0, help="seed of the synthetic sample")
    ap.add_argument("--frames", type=int, nargs="*", help="only render these frames (1-based)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    P = make_problem(args.seed)
    for step in args.frames or range(1, N_FRAMES + 1):
        fig = draw_frame(step, P, args.dpi)
        path = args.outdir / f"frame_{step:02d}.{args.format}"
        fig.savefig(path, facecolor=BEIGE)
        plt.close(fig)
        print(path)
    if not args.frames:
        try:
            rel = args.outdir.resolve().relative_to(REPO / "slides").as_posix()
        except ValueError:
            rel = args.outdir.resolve().as_posix()
        snippet = args.outdir / "slides_snippet.md"
        snippet.write_text(slides_snippet(rel, args.format))
        print(snippet)


if __name__ == "__main__":
    main()
