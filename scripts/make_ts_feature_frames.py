# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy"]
# ///
"""Step-by-step diagram of the time series feature engineering for tabular models.

Generates one image per animation frame. Each frame adds one element to the
previous one and all frames share the same layout, so that going from one slide
to the next looks like an animation (see the generated ``slides_snippet.md``).

The diagram illustrates what ``add_features`` in
``electricity_load_forecasting/pipeline.py`` (skore-example-electricity-load)
does for a single prediction time ``p`` and a single horizon ``h = 12 h``, with
the three optional feature groups used by the best models of round 2
(``--feature_groups load_diffs load_same_hour temp_diffs``): calendar features,
load lags, load differences, rolling median / IQR, weather forecasts and
national temperature changes. The resulting feature table is fed to HGB,
TabICL or Kumo-Tabular. The load and temperature curves are synthetic (the
shapes are realistic, the values are not).

Usage::

    uv run scripts/make_ts_feature_frames.py            # PNG frames in slides/assets/ts_features
    uv run scripts/make_ts_feature_frames.py --format svg
    uv run scripts/make_ts_feature_frames.py --frames 4 5 --outdir /tmp/frames
"""

from __future__ import annotations

import argparse
import datetime as dt
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
BLUE, ORANGE = "#040524", "#ff7900"
GRAY = "#5b5b66"
FORBID = "#c0392b"
# one colour per feature group, reused in the timeline and in the table
CAL, LAG, LDIF, ROLL, WEA, TDIF = "#7a4fb5", ORANGE, "#2e8b57", "#d6336c", "#1a8fc4", "#0b3d91"
GROUPS = ("cal", "lag", "ldiff", "roll", "wea", "tdiff")  # order of introduction
GROUP_COLOR = {"cal": CAL, "lag": LAG, "ldiff": LDIF, "roll": ROLL, "wea": WEA, "tdiff": TDIF}
GROUP_LABEL = {
    "cal": "calendar",
    "lag": "load lags",
    "ldiff": "load diffs",
    "roll": "rolling stats",
    "wea": "weather",
    "tdiff": "temp diffs",
}


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
# What the pipeline does, for the illustrated case
# --------------------------------------------------------------------------
H = 12  # horizon (hours)
P = dt.datetime(2025, 3, 12, 6)  # prediction time p (a Wednesday, 06:00)
N_CITIES = 10
N_ROLLS = 8  # 3 h windows covering the 24 h before the target time
N_CAL = 5


def load_lags(h):
    """Load lags of ``add_lagged_features`` with ``load_same_hour``."""
    return sorted(set(range(h, 24)) | {24, 48, 168} | {h + 24, h + 168})


def load_diff_pairs(h):
    """Pairs (a, b) of the load differences lag_a - lag_b.

    ``load_diffs`` gives the consecutive differences and lag_h minus the lags
    24, 48, 168; ``load_same_hour`` adds lag_h minus lag_{h+24}, lag_{h+168}.
    """
    pairs = [(k, k + 1) for k in range(h, 23)]
    pairs += [(h, k) for k in (24, 48, 168, h + 24, h + 168) if k != h]
    return list(dict.fromkeys(pairs))


def temp_diff_count(h):
    """National temperature changes over 24 h, 168 h and h, plus 2 anomalies."""
    return len({24, 168, h}) + 2


def feature_counts(h):
    """Number of features per group, with the three round-2 feature groups."""
    return {
        "cal": N_CAL,
        "lag": len(load_lags(h)),
        "ldiff": len(load_diff_pairs(h)),
        "roll": len({h, 24}) * 2 * 2,  # (lags) x (24 h, 168 h) x (median, IQR)
        "wea": N_CITIES * (1 + N_ROLLS),  # temperature at t + 8 rolling means
        "tdiff": temp_diff_count(h),
    }


def done_counts(step):
    """Features of each group already introduced at ``step``."""
    final = feature_counts(H)
    now = dict.fromkeys(GROUPS, 0)
    if step >= 12:
        return final
    if step >= 3:
        now["cal"] = final["cal"]
    if step >= 4:
        now["lag"] = 12  # lag_12 ... lag_23
    if step >= 5:
        now["lag"] = final["lag"]
    if step >= 6:
        now["ldiff"] = 11  # lag_k - lag_{k+1}, k = 12 ... 22
    if step >= 7:
        now["ldiff"] = final["ldiff"]
    if step >= 8:
        now["roll"] = final["roll"]
    if step >= 9:
        now["wea"] = N_CITIES
    if step >= 10:
        now["wea"] = final["wea"]
    if step >= 11:
        now["tdiff"] = final["tdiff"]
    return now


# --------------------------------------------------------------------------
# Synthetic data, indexed by hours relative to p (x = 0 is "now")
# --------------------------------------------------------------------------
class Series:
    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        self.x = np.arange(-260, 61)
        times = [P + dt.timedelta(hours=int(x)) for x in self.x]
        hod = np.array([t.hour for t in times], dtype=float)
        dow = np.array([t.weekday() for t in times])

        def bump(c, w):
            return np.exp(-0.5 * (((hod - c + 12) % 24 - 12) / w) ** 2)

        daily = 9 * bump(9.5, 2.3) + 11 * bump(19.5, 2.2) - 8 * bump(4, 3) + 4 * bump(14, 3)
        weekend = np.where(dow == 5, -6.0, np.where(dow == 6, -9.0, 0.0))
        weekend = np.convolve(np.pad(weekend, (4, 4), mode="edge"), np.ones(9) / 9, "valid")
        days = self.x / 24
        temp = 7 + 3 * np.sin(2 * np.pi * days / 6.5) + 4 * np.sin(2 * np.pi * (hod - 9) / 24)
        noise = np.zeros(len(self.x))
        for i in range(1, len(noise)):
            noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.5)
        self.load = 1000 * (55 + daily + weekend + 0.8 * (10 - temp) + noise)  # MW
        self.temps = np.array(
            [
                temp + off + np.convolve(rng.normal(0, 0.5, len(temp)), np.ones(4) / 4, "same")
                for off in (0.0, 2.8, -2.2)
            ]
        )
        # stand-in for the national mean over the cities (only 3 are drawn)
        self.nat = self.temps.mean(axis=0)

    def at(self, x):
        return P + dt.timedelta(hours=int(x))

    def L(self, x):  # load in MW at relative hour x
        return self.load[int(x) + 260]

    def L_gw(self, xs):
        return self.load[np.asarray(xs, dtype=int) + 260] / 1000

    def T(self, city, x):
        return self.temps[city][int(x) + 260]

    def N(self, x):  # national mean temperature at relative hour x
        return self.nat[int(x) + 260]

    def window(self, end, width):  # values over the window (end - width, end]
        return self.load[end - width + 1 + 260 : end + 1 + 260]

    def national_window(self, end, width):
        return self.nat[end - width + 1 + 260 : end + 1 + 260]


# --------------------------------------------------------------------------
# Frame definitions: title, subtitle, speaker notes
# --------------------------------------------------------------------------
FRAMES = [
    (
        "Start from the hourly load",
        "One value per hour, resampled on a regular grid. Daily and weekly cycles dominate.",
        "Raw material: the national load, one row per hour (`resample`). Point at the daily "
        "double peak and at the weekend dip: that is what the lags will exploit.",
    ),
    (
        "Pick a prediction time and a horizon",
        "At time p only the past is known. Target: the load at t = p + h, here h = 12 h.",
        "One row of the training table = one prediction time p. The target is "
        "`load_mw.shift(-h)`. We repeat the construction for h = 1, 12 and 24 h.",
    ),
    (
        "Calendar features of the target time",
        "Calendar information about t is known in advance: no leakage possible.",
        "5 features computed on the *target* time, in Europe/Paris time: hour of day, day of "
        "week, day of year, year and public holiday flag.",
    ),
    (
        "Recent load lags",
        "Lag k = load at t - k. Only lags k >= h are observed at p: here lag_12 to lag_23.",
        "For h = 12 the most recent usable observation is p itself (= lag_12). Lags 1 to 11 "
        "would peek into the future, so they are not built. This is why the features are "
        "built per horizon.",
    ),
    (
        "Same clock hour, earlier days",
        "Lags 24 h, 48 h and 168 h (same hour as t), and 36 h and 180 h (same hour as p).",
        "Exploits the daily and weekly seasonality. `load_same_hour` adds lags 36 and 180: "
        "the same clock hour as p, one day and one week before. 17 load lags in total for h = 12.",
    ),
    (
        "Hourly changes of the load",
        "Differences between consecutive lags: the slope of the load over the 12 hours up to p.",
        "`load_diffs`: lag_k - lag_(k+1) for k = h ... 22, so 11 features for h = 12. A tree "
        "cannot subtract two columns itself, so the differences are given explicitly.",
    ),
    (
        "Load now versus earlier loads",
        "lag_12 is load(p): its difference with the lags 24, 36, 48, 168 and 180 h.",
        "`load_diffs` adds lag_h - lag_24, lag_48, lag_168 and `load_same_hour` adds "
        "lag_h - lag_36, lag_180: 5 more features, 16 load differences in total.",
    ),
    (
        "Rolling median and IQR",
        "Typical level and spread of the load over the last day and the last week.",
        "median and IQR over 24 h and 168 h windows ending at p and at t - 24 h: "
        "2 stats x 2 widths x 2 end points = 8 features.",
    ),
    (
        "Weather forecasts",
        "Temperature forecast valid at t, for each of 10 cities (a forecast issued 24 h earlier).",
        "Weather *forecasts* rather than observations: a forecast issued at least 24 h before "
        "the valid time is assumed available at p (day-1 forecast for h <= 18 h, day-2 above). "
        "One feature per city, 3 of 10 shown.",
    ),
    (
        "Rolling temperature means",
        "Mean temperature over eight 3 h windows covering the 24 h before t, for each city.",
        "9 features per city (temperature at t + 8 rolling means) x 10 cities = 90 weather "
        "features: the biggest group by far.",
    ),
    (
        "National temperature changes",
        "National mean temperature: changes over 24 h, 168 h and h hours, and anomalies.",
        "`temp_diffs`: change of the national mean temperature over 24 h, 168 h and h, and "
        "anomalies against its 24 h and 168 h means. Windows end at t and only forecasts valid "
        "by t are used. 5 features that combine the cities, which a tree cannot do itself.",
    ),
    (
        "One feature row per prediction time",
        "141 numbers describe (p, h = 12 h); the target y is the load at t.",
        "Column names are abbreviated. `prediction_time` and `target_time` are dropped and "
        "everything is converted to floats: the tabular models see no timestamp, the time "
        "structure only lives in the features. These are the round-2 feature groups.",
    ),
    (
        "Slide p forward: a plain tabular dataset",
        "One row for every hour; one such table for each horizon.",
        "Number of features per horizon: 163 (h = 1), 141 (h = 12), 110 (h = 24). Without the "
        "three feature groups: 129, 118, 102. Evaluation uses walk-forward CV with a gap "
        "between the train and test periods.",
    ),
    (
        "Same table, four tabular models",
        "The four models consume the same (X, y); only the model changes.",
        "HGB quantile: one model per quantile. HGB mean: one model for the mean. TabICL and "
        "Kumo-Tabular (TFMs): `fit` just stores (X, y); prediction runs the pretrained transformer "
        "with the training table as context, and returns quantiles or the mean. Kumo returns 999 "
        "quantiles, we keep 5/50/95%.",
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
    bg.text(LEFT, 6.95, title, family=SERIF, size=26, color=BLUE, va="top")
    bg.add_patch(Rectangle((LEFT, 6.42), 0.6, 0.05, color=ORANGE, lw=0))
    bg.text(LEFT, 6.3, subtitle, size=14, color=GRAY, va="top")
    return fig, bg


def axes_in(fig, left, bottom, width, height, **kw):
    return fig.add_axes([left / W, bottom / HT, width / W, height / HT], **kw)


def box(bg, x, y, w, h, fc, ec=None, lw=1.2, alpha=1.0, ls="-", radius=0.06, z=1):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=fc,
        ec=ec if ec else "none",
        lw=lw,
        alpha=alpha,
        ls=ls,
        zorder=z,
    )
    bg.add_patch(p)
    return p


def bracket(ax, x0, x1, y, color, cap, lw=3.0, alpha=1.0):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, alpha=alpha, solid_capstyle="butt")
    for x in (x0, x1):
        ax.plot([x, x], [y - cap, y + cap], color=color, lw=lw, alpha=alpha)


def caption(bg, lines):
    box(bg, LEFT, 1.55, WIDTH, 1.05, "white", BEIGE_DARK, lw=1.5, alpha=0.8)
    for i, line in enumerate(lines):
        bg.text(LEFT + 0.2, 2.35 - 0.33 * i, line, family=MONO, size=12.5, color=BLUE, va="top")


def fmt_gw(v):
    """Load in MW shown in GW with one decimal (keeps the table cells narrow)."""
    return f"{v / 1000:.1f}"


def fmt_signed_gw(v):
    return f"{v / 1000:+.1f}"


# --------------------------------------------------------------------------
# Frames 1 to 8: load timeline, frames 9 to 11: temperature panel
# --------------------------------------------------------------------------
def feature_bar(bg, step):
    """Bottom bar: the feature row being assembled (empty placeholders at first)."""
    final = feature_counts(H)
    now = done_counts(step)
    # bar width grows with the square root of the number of features in each group
    weights = {g: np.sqrt(n) for g, n in final.items()}
    gap = 0.07
    unit = (WIDTH - (len(GROUPS) - 1) * gap) / sum(weights.values())
    total = sum(now.values())
    bg.text(
        LEFT,
        1.3,
        f"Feature row for (p, h = {H} h): {total} features" if total else "Feature row for (p, h = 12 h)",
        size=13.5,
        color=BLUE,
        va="bottom",
        weight="bold",
    )
    x = LEFT
    for g in GROUPS:
        w = weights[g] * unit
        c = GROUP_COLOR[g]
        box(bg, x, 0.36, w, 0.46, "none", c, lw=1.4, ls=(0, (3, 2)), alpha=0.6)
        if now[g]:
            frac = now[g] / final[g]
            box(bg, x, 0.36, max(w * frac, 0.06), 0.46, c, radius=0.05)
            bg.text(
                x + w / 2, 0.9, f"{GROUP_LABEL[g]} {now[g]}", size=11.5, color=c,
                ha="center", va="bottom", weight="bold",
            )
        x += w + gap


def draw_timeline(step, S, dpi):
    title, subtitle, _ = FRAMES[step - 1]
    fig, bg = new_canvas(dpi, title, subtitle)

    ax = axes_in(fig, LEFT + 0.3, 2.95, WIDTH - 0.3, 3.0)
    ax.patch.set_alpha(0)
    xs = S.x[(S.x >= -204) & (S.x <= 42)]
    gw = S.L_gw(xs)
    dmin, dmax = gw.min(), gw.max()
    row_gap = 6.0
    band_top = dmin - 5
    ymin, ymax = band_top - 3 * row_gap - 4, dmax + 20
    ax.set_xlim(-204, 42)
    ax.set_ylim(ymin, ymax)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.set_yticks([50, 60, 70])
    ax.set_yticklabels(["50", "60", "70"], size=11, color=GRAY)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, colors=GRAY, labelsize=11)
    midnights = [x for x in range(-204, 43) if S.at(x).hour == 0]
    ax.set_xticks(midnights)
    ax.set_xticklabels([S.at(x).strftime("%a %-d") for x in midnights])
    for x in midnights:
        ax.axvline(x, color=BEIGE_DARK, lw=1, zorder=0)

    # weekends
    for d in ("Sat", "Sun"):
        xm = [x for x in midnights if S.at(x).strftime("%a") == d]
        for x in xm:
            ax.axvspan(x, x + 24, color="#d9e4f0", alpha=0.6, lw=0, zorder=0)
    if step == 1:
        ax.text(-102 + 24, dmax + 3, "weekend", ha="center", size=12, color=GRAY, style="italic")

    # load curve: the future is unknown from step 2 on
    past, fut = xs <= 0, xs >= 0
    if step == 1:
        ax.plot(xs, gw, color=BLUE, lw=2.4)
    else:
        ax.plot(xs[past], gw[past], color=BLUE, lw=2.4)
        ax.plot(xs[fut], gw[fut], color=GRAY, lw=1.6, ls=(0, (1.5, 2.5)), alpha=0.7)
        ax.axvspan(0, 42, color=BEIGE_DARK, alpha=0.45, lw=0, zorder=0)
        ax.axvline(0, color=BLUE, lw=1.6, ls="--", zorder=1)
        ax.text(0, ymax - 1.5, "p = now", ha="center", va="top", size=13,
                weight="bold", color=BLUE, bbox=dict(fc=BEIGE, ec="none", pad=1.5))

    def a(step_of_group):  # newest group at full strength, older ones faded
        return 1.0 if step == step_of_group else 0.4

    # target
    if step >= 2:
        yt = S.L(H) / 1000
        ax.plot([H], [yt], "o", ms=15, mfc="white", mec=BLUE, mew=2.2, zorder=6)
        ax.text(H, yt, "?", ha="center", va="center", size=12, weight="bold", color=BLUE, zorder=7)
        ax.text(H + 3.5, yt, "target\ny = load(t)\nt = p + h", va="center", size=11.5,
                color=BLUE, linespacing=1.15, zorder=6,
                bbox=dict(fc=BEIGE, ec="none", alpha=0.92, pad=1.5))

    # step 3: calendar chips
    if step == 3:
        t = S.at(H)
        txt = (f"t = {t:%a %H:%M}   →   hour {t.hour} · weekday {t.isoweekday()} · "
               f"day {t.timetuple().tm_yday} · {t.year} · holiday: no")
        ax.text(42, dmax + 9, txt, ha="right", va="center", size=12, color=BLUE,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=CAL, lw=2))
        ax.plot([H, H], [S.L(H) / 1000 + 2.5, dmax + 6.4], color=CAL, lw=1.6, ls=":")

    # steps 4+: unavailable lags and recent lags (k = H .. 23 <-> x = H - k)
    if step >= 4:
        al = 1.0 if step == 4 else 0.6
        ax.add_patch(Rectangle((0, ymin), H, ymax - ymin, fc=FORBID, alpha=0.10 * al,
                               ec=FORBID, hatch="///", lw=0, zorder=1))
        ax.text(H / 2, (ymin + ymax) / 2, "lag_1…11 unknown", rotation=90, ha="center",
                va="center", size=11.5, color=FORBID, alpha=al)
        xk = [H - k for k in range(H, 24)]
        ax.plot(xk, S.L_gw(xk), "o", ms=7.5, color=LAG, mec="white", mew=0.8,
                alpha=a(4), zorder=5)
        if step == 4:
            bracket(ax, min(xk), max(xk), dmax + 4, LAG, 1.0)
            ax.text(min(xk) - 2, dmax + 4, "lag_12 … lag_23", ha="right", va="center",
                    size=12.5, color=LAG, weight="bold")

    # step 5+: same clock hour lags; the offsets stagger the labels
    if step >= 5:
        label_height = {24: 5.5, 48: 5.5, 168: 5.5, 36: 10.5, 180: 10.5}
        for k, dy in label_height.items():
            x = H - k
            ax.axvline(x, color=LAG, lw=1.2, ls=":", alpha=0.7 * a(5), zorder=1)
            ax.plot([x], [S.L(x) / 1000], "o", ms=11, color=LAG, mec="white", mew=1.2,
                    alpha=a(5), zorder=5)
            if step == 5:
                ax.text(x, S.L(x) / 1000 + dy, f"lag_{k}", ha="center", size=12.5,
                        color=LAG, weight="bold",
                        bbox=dict(fc=BEIGE, ec="none", alpha=0.9, pad=1.2))
        ax.axvline(H, color=LAG, lw=1.2, ls=":", alpha=0.7 * a(5), zorder=1)

    # step 6+: consecutive load differences lag_k - lag_{k+1}
    if step >= 6:
        for k in range(H, 23):
            ax.plot([H - k, H - k - 1], [S.L(H - k) / 1000, S.L(H - k - 1) / 1000],
                    color=LDIF, lw=3.5, alpha=a(6), solid_capstyle="round", zorder=4)
        if step == 6:
            xk = [H - k for k in range(H, 24)]
            bracket(ax, min(xk), max(xk), dmax + 4, LDIF, 1.0)
            ax.text(min(xk) - 2, dmax + 4, "lag_k − lag_k+1", ha="right", va="center",
                    size=12.5, color=LDIF, weight="bold")

    # step 7+: differences between lag_12 (= load(p)) and the reference lags
    if step >= 7:
        base = S.L(0) / 1000
        ref = (24, 36, 48, 168, 180)
        for k in ref:
            x = H - k
            y = S.L(x) / 1000
            ax.plot([x, x], [base, y], color=LDIF, lw=2.4, alpha=a(7), zorder=5)
            ax.plot([x], [y], "o", ms=8, color=LDIF, mec="white", mew=1, alpha=a(7), zorder=6)
            if step == 7:
                ax.text(x, max(base, y) + 1.8, str(k), ha="center", size=12.5,
                        color=LDIF, weight="bold",
                        bbox=dict(fc=BEIGE, ec="none", alpha=0.9, pad=1.0))
        ax.plot([H - max(ref), 0], [base, base], color=LDIF, lw=1.1, ls=":",
                alpha=0.8 * a(7), zorder=4)

    # step 8+: rolling windows
    if step >= 8:
        rows = [(168, 0), (24, 0), (168, -12), (24, -12)]  # (width, end), end = t - lag
        for i, (width, end) in enumerate(rows):
            y = band_top - i * row_gap
            bracket(ax, end - width, end, y, ROLL, 1.1, alpha=a(8))
            ax.text(end - width - 2, y, f"{width} h", ha="right", va="center", size=11.5,
                    color=ROLL, alpha=a(8))
        for end, name in ((0, "p"), (-12, "t − 24 h")):
            ax.plot([end, end], [ymin, band_top + 1.5], color=ROLL, lw=1.2, ls=":", alpha=a(8))
            ax.text(end, ymin + 0.3, name, ha="center", va="bottom", size=11.5, color=ROLL,
                    alpha=a(8), bbox=dict(fc=BEIGE, ec="none", pad=1))
        if step == 8:
            ax.text(-204, band_top + row_gap * 0.7, "median and IQR of each window", size=12,
                    color=ROLL, weight="bold", va="bottom")

    ax.text(-203, ymax - 1.5, "Electricity load (GW)", size=12, color=GRAY, va="top")

    # --- lower area: caption (steps 1-8) or temperature panel (steps 9-11)
    if step <= 8:
        caption(bg, CAPTIONS[step])
    else:
        draw_temperature_panel(fig, S, step)
    if step >= 2:
        feature_bar(bg, step)
    return fig


CAPTIONS = {
    1: [
        "resample(load_mw_history)          # 1 row per hour, missing hours -> explicit nulls",
        "all load features below are computed from this single series",
    ],
    2: [
        "X : one row per prediction time p       y : load_mw.shift(-h),  t = p + h",
        "one model per horizon h in (1, 12, 24)  -> predicts the 5 %, 50 % and 95 % quantiles of y",
    ],
    3: [
        "cal_hour_of_day  cal_day_of_week  cal_day_of_year  cal_year  cal_is_holiday",
        "computed on the target time t (Europe/Paris)  ->  5 features",
    ],
    4: [
        "lag_k = load(t - k)   for k = h ... 23          (h = 12: lag_12 ... lag_23)",
        "lags k < h do not exist yet at time p: they would leak the future",
    ],
    5: [
        "lag_24  lag_48  lag_168          # same clock hour as t: 1 day, 2 days, 1 week before",
        "lag_36  lag_180                  # same clock hour as p: 1 day, 1 week before  -> 17 lags",
    ],
    6: [
        "lag_k - lag_k+1   for k = 12 ... 22          # hourly changes of the load (load_diffs)",
        "11 features: the local slope of the load in the 12 hours up to p",
    ],
    7: [
        "lag_12 - lag_k    for k = 24, 36, 48, 168, 180           # load(p) vs. earlier loads",
        "-> 5 more features: 11 + 5 = 16 load differences in total",
    ],
    8: [
        "lag_{12|24}_width_{24|168}_{median|iqr}      # windows (t-lag-width, t-lag]",
        "2 end points x 2 widths x 2 statistics  ->  8 features (the end points are lag_h and lag_24)",
    ],
}


def draw_temperature_panel(fig, S, step):
    ax = axes_in(fig, LEFT + 0.3, 1.55, WIDTH - 0.3, 1.05)
    ax.patch.set_alpha(0)
    ax.set_xlim(-204, 42)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.set_yticks([])
    ax.set_xticks([])
    xs = S.x[(S.x >= -204) & (S.x <= 42)]
    idx = xs + 260
    shades = [WEA, "#3ba7d9", "#0f6a96"]
    lo = S.temps[:, idx].min() - 1.5
    hi = S.temps[:, idx].max() + 5
    ax.set_ylim(lo, hi)
    for x in range(-204, 43):
        if S.at(x).hour == 0:
            ax.axvline(x, color=BEIGE_DARK, lw=1, zorder=0)
    ax.axvline(0, color=BLUE, lw=1.4, ls="--", zorder=1)
    ax.axvspan(0, 42, color=BEIGE_DARK, alpha=0.45, lw=0, zorder=0)
    city_alpha = 0.3 if step == 11 else (0.55 if step == 10 else 0.9)
    for c in range(3):
        ax.plot(xs, S.temps[c][idx], color=shades[c], lw=1.6, alpha=city_alpha)

    if step == 11:
        ax.plot(xs, S.nat[idx], color=TDIF, lw=2.6, zorder=3)
        ax.text(-203, hi - 0.3, "National mean temperature (average of the 3 cities drawn)",
                size=12, color=TDIF, va="top", weight="bold")
        # the temperatures at t, t - h, t - 24 h and t - 168 h
        for x, name in ((H, "t"), (0, "p"), (H - 24, "t−24"), (H - 168, "t−168")):
            ax.plot([x, x], [lo + 0.4, S.N(x)], color=TDIF, lw=1, ls=":", zorder=2)
            ax.plot([x], [S.N(x)], "o", ms=7, color=TDIF, mec="white", mew=1, zorder=5)
            ax.text(x, lo + 0.5, name, ha="center", va="bottom", size=10.5, color=TDIF,
                    bbox=dict(fc=BEIGE, ec="none", pad=0.5))
        # anomalies: the means over (t - 24 h, t] and (t - 168 h, t]
        for width, lab in ((24, "mean 24 h"), (168, "mean 168 h")):
            level = S.national_window(H, width).mean()
            ax.plot([H - width + 1, H], [level, level], color=TDIF, lw=1.4, ls="--", zorder=4)
            ax.text(H - width + 1, level, lab, ha="left", va="bottom", size=10.5, color=TDIF,
                    bbox=dict(fc=BEIGE, ec="none", alpha=0.9, pad=0.5))
        return

    ax.text(-203, hi - 0.3, "Temperature forecasts (3 of 10 cities shown)", size=12,
            color=WEA, va="top", weight="bold")
    if step == 9:
        ax.axvline(H, color=WEA, lw=1.2, ls=":")
        for c in range(3):
            ax.plot([H], [S.temps[c][H + 260]], "o", ms=9, color=shades[c], mec="white", mew=1)
        ax.text(H - 3, hi - 0.3, "temperature_2m at t\n(1 feature per city)", ha="right", va="top",
                size=11.5, color=WEA, linespacing=1.15)
    else:
        for j in range(N_ROLLS):
            x0 = H - 24 + 3 * j
            ax.axvspan(x0, x0 + 3, color=WEA, alpha=0.16 if j % 2 else 0.30, lw=0, zorder=1)
        ax.text(H - 24 - 2, hi - 0.3, "8 × 3 h means over the 24 h before t\n(8 features per city)",
                ha="right", va="top", size=11.5, color=WEA, linespacing=1.15)
        ax.axvline(H, color=WEA, lw=1.2, ls=":")


# --------------------------------------------------------------------------
# Frames 12 and 13: feature table, frame 14: models
# --------------------------------------------------------------------------
COLS = {  # key: (x, width), columns separated by 0.05 in
    "idx": (0.45, 0.95),
    "cal": (1.45, 0.80),
    "lag": (2.30, 1.80),
    "ldiff": (4.15, 1.80),
    "roll": (6.00, 1.80),
    "wea": (7.85, 2.00),
    "tdiff": (9.90, 1.35),
    "y": (11.30, 0.95),
}
SUBHEAD = {
    "cal": ["hr", "dow", "…"],
    "lag": ["lag_12", "lag_13", "…"],
    "ldiff": ["Δ12-13", "Δ12-24", "…"],
    "roll": ["iqr_24", "med_168", "…"],
    "wea": ["T_paris", "T_lyon", "…"],
    "tdiff": ["Δ24", "anom24"],
}
HEAD_Y, HEAD_H, SUB_Y = 5.4, 0.42, 5.0
ROW_H = 0.42


def table_header(bg, counts):
    for key, (x, w) in COLS.items():
        if key == "idx":
            bg.text(x + w / 2, HEAD_Y + HEAD_H / 2, "prediction\ntime p", ha="center", va="center",
                    size=11.5, color=GRAY, linespacing=1.1)
            continue
        color = BLUE if key == "y" else GROUP_COLOR[key]
        if key == "y":
            label, size = "target y", 12
        elif w < 1.0:  # narrow column: two lines
            label, size = f"{GROUP_LABEL[key].split()[0]}\n{counts[key]}", 10.5
        else:
            label, size = f"{GROUP_LABEL[key]} · {counts[key]}", 12
        box(bg, x, HEAD_Y, w, HEAD_H, color)
        bg.text(x + w / 2, HEAD_Y + HEAD_H / 2, label, ha="center", va="center", size=size,
                color="white", weight="bold", linespacing=1.0)
        names = SUBHEAD.get(key, ["load(t)"])
        for j, n in enumerate(names):
            cw = w / len(names)
            bg.text(x + cw * (j + 0.5), SUB_Y + 0.12, n, ha="center", va="center",
                    family=MONO, size=10, color=color)


def row_values(S, xp):
    t = xp + H
    window = S.window(xp, 24)
    q75, q25 = np.percentile(window, [75, 25])
    return {
        "idx": S.at(xp).strftime("%a %H:%M"),
        "cal": [str(S.at(t).hour), str(S.at(t).isoweekday()), "…"],
        "lag": [fmt_gw(S.L(t - 12)), fmt_gw(S.L(t - 13)), "…"],
        "ldiff": [fmt_signed_gw(S.L(t - 12) - S.L(t - 13)),
                  fmt_signed_gw(S.L(t - 12) - S.L(t - 24)), "…"],
        "roll": [fmt_gw(q75 - q25), fmt_gw(np.median(S.window(xp, 168))), "…"],
        "wea": [f"{S.T(0, t):.1f}", f"{S.T(1, t):.1f}", "…"],
        "tdiff": [f"{S.N(t) - S.N(t - 24):+.1f}",
                  f"{S.N(t) - S.national_window(t, 24).mean():+.1f}"],
        "y": [fmt_gw(S.L(t))],
    }


def table_row(bg, S, xp, y, highlight):
    vals = row_values(S, xp)
    if highlight:
        box(bg, 0.38, y - 0.03, 11.98, ROW_H + 0.06, "white", ORANGE, lw=2.4, z=0)
    bg.text(COLS["idx"][0] + 0.05, y + ROW_H / 2, vals["idx"], family=MONO, size=11, va="center",
            color=BLUE if highlight else GRAY)
    for key in COLS:
        if key == "idx":
            continue
        x, w = COLS[key]
        cells = vals[key]
        for j, v in enumerate(cells):
            cw = w / len(cells)
            bg.text(x + cw * (j + 0.5), y + ROW_H / 2, v, ha="center", va="center", family=MONO,
                    size=11, color=BLUE if highlight else GRAY)
    bg.plot([0.45, 12.35], [y - 0.03, y - 0.03], color=BEIGE_DARK, lw=1)


def bullets(bg, y, lines):
    for i, line in enumerate(lines):
        bg.text(LEFT + 0.1, y - 0.42 * i, f"•  {line}", size=14, color=BLUE, va="top")


def draw_table(step, S, dpi):
    title, subtitle, _ = FRAMES[step - 1]
    fig, bg = new_canvas(dpi, title, subtitle)
    counts = feature_counts(H)
    table_header(bg, counts)
    if step == 12:
        y = SUB_Y - 0.15 - ROW_H
        table_row(bg, S, 0, y, True)
        x0 = COLS["cal"][0]
        x1 = COLS["tdiff"][0] + COLS["tdiff"][1]
        by = y - 0.2
        bg.plot([x0, x1], [by, by], color=BLUE, lw=2)
        bg.plot([x0, x0], [by, by + 0.1], color=BLUE, lw=2)
        bg.plot([x1, x1], [by, by + 0.1], color=BLUE, lw=2)
        bg.text((x0 + x1) / 2, by - 0.12, f"X: {sum(counts.values())} numeric columns", ha="center",
                va="top", size=13.5, color=BLUE, weight="bold")
        bullets(bg, 3.05, [
            "one row = one prediction time p (here Wed 12 Mar, 06:00)",
            "the time structure only lives in the lag, difference and rolling features",
            "prediction_time and target_time are dropped: the models only see floats",
            "load values in GW (y: load at t); temperatures in °C",
        ])
    else:
        rows = [-2, -1, 0, 1, 2]
        y = SUB_Y - 0.15 - ROW_H
        for xp in rows:
            table_row(bg, S, xp, y, xp == 0)
            y -= ROW_H + 0.03
        for key in COLS:
            x, w = COLS[key]
            bg.text(x + w / 2, y + 0.12, "⋮", ha="center", va="center", size=16, color=GRAY)
        c = {h: sum(feature_counts(h).values()) for h in (1, 12, 24)}
        bullets(bg, 1.75, [
            f"one table per horizon: {c[1]} / {c[12]} / {c[24]} features for h = 1 / 12 / 24 h",
            "y is only known for past rows: train on history, predict forward (walk-forward CV)",
            "load values in GW; temperatures in °C",
        ])
    return fig


def arrow(bg, x0, y0, x1, y1, color=GRAY):
    bg.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, shrinkA=0, shrinkB=0))


def draw_models(step, S, dpi):
    title, subtitle, _ = FRAMES[step - 1]
    fig, bg = new_canvas(dpi, title, subtitle)
    counts = feature_counts(H)

    # input table (mini)
    tx, ty, tw, th = 0.45, 2.7, 2.4, 2.1
    box(bg, tx, ty, tw, th, "white", BEIGE_DARK, lw=1.5)
    cx = tx + 0.15
    full = tw - 0.3 - 0.45
    tot = sum(counts.values())
    for g in GROUPS:
        w = full * counts[g] / tot
        bg.add_patch(Rectangle((cx, ty + 0.3), max(w, 0.03), th - 0.6, fc=GROUP_COLOR[g], lw=0, alpha=0.9))
        cx += max(w, 0.03)
    bg.add_patch(Rectangle((cx + 0.1, ty + 0.3), 0.3, th - 0.6, fc=BLUE, lw=0))
    bg.text(tx + tw / 2, ty + th - 0.14, "X, y", ha="center", va="center", size=13, color=BLUE, weight="bold")
    bg.text(tx + tw / 2, ty + 0.14, f"n rows × {tot}", ha="center", va="center", size=11.5, color=GRAY)

    models = [
        ("HGB quantile", "scikit-learn", "3 boosted-tree models,\none per quantile", BLUE),
        ("HGB mean", "scikit-learn", "1 boosted-tree model,\nsquared error: the mean", BLUE),
        ("TabICL", "tabular foundation model", "fit: stores (X, y)\npredict: quantiles or mean, (X, y) as context", ORANGE),
        ("Kumo-Tabular", "tabular foundation model", "same recipe, 999 quantiles,\nwe keep the 3 we need", ORANGE),
    ]
    bx, bw, bh, gap = 4.0, 4.5, 1.05, 0.18
    top = 5.75
    for i, (name, family, desc, color) in enumerate(models):
        y = top - bh - i * (bh + gap)
        box(bg, bx, y, bw, bh, "white", color, lw=2.4)
        bg.text(bx + 0.2, y + bh - 0.14, name, size=15, color=BLUE, va="top", weight="bold")
        bg.text(bx + bw - 0.2, y + bh - 0.18, family, size=11, color=GRAY, va="top", ha="right")
        bg.text(bx + 0.2, y + 0.12, desc, size=12, color=BLUE, va="bottom", linespacing=1.25)
        arrow(bg, tx + tw + 0.05, ty + th / 2, bx - 0.05, y + bh / 2)
        arrow(bg, bx + bw + 0.05, y + bh / 2, 9.95, 3.75 + (1.5 - i) * 0.45)

    # output
    ox, oy, ow, oh = 10.0, 2.7, 2.35, 2.1
    box(bg, ox, oy, ow, oh, "white", BEIGE_DARK, lw=1.5)
    bg.text(ox + ow / 2, oy + oh - 0.16, "load(t)", ha="center", va="top", size=13, color=BLUE, weight="bold")
    bg.text(ox + ow / 2, oy + 0.14, "q5 · q50 · q95", ha="center", va="bottom", size=11.5, color=GRAY)
    mx = ox + ow / 2
    bg.plot([mx, mx], [oy + 0.65, oy + 1.55], color=BLUE, lw=2)
    for yy, ww in ((0.65, 0.3), (1.1, 0.45), (1.55, 0.3)):
        bg.plot([mx - ww / 2, mx + ww / 2], [oy + yy, oy + yy], color=BLUE, lw=3)
    bg.text(0.45, 0.62, "× 3 horizons (1, 12, 24 h): one independent model per horizon and method",
            size=13, color=GRAY, va="center")
    return fig


def render(step, S, dpi):
    if step <= 11:
        return draw_timeline(step, S, dpi)
    if step <= 13:
        return draw_table(step, S, dpi)
    return draw_models(step, S, dpi)


# --------------------------------------------------------------------------
def slides_snippet(outdir, fmt, rel):
    """remark.js markdown: one slide per frame, the frame is the slide background."""
    # Generated by scripts/make_ts_feature_frames.py: paste into slides/index.html.
    out = []
    for i, (title, _, notes) in enumerate(FRAMES, start=1):
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
    ap.add_argument("--outdir", type=Path, default=REPO / "slides/assets/ts_features")
    ap.add_argument("--format", choices=["png", "svg"], default="png")
    ap.add_argument("--dpi", type=int, default=150, help="150 dpi gives 1920x1080 px")
    ap.add_argument("--frames", type=int, nargs="*", help="only render these frames (1-based)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    S = Series()
    for step in args.frames or range(1, N_FRAMES + 1):
        fig = render(step, S, args.dpi)
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
        snippet.write_text(slides_snippet(args.outdir, args.format, rel))
        print(snippet)


if __name__ == "__main__":
    main()
