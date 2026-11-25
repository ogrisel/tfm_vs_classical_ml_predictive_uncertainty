"""Paired differences between the models of the comparison, from ``raw_results``.

Reproduces the numbers of the bonus slides:

- paired pinball-loss differences (mean of the 5% and 95% levels, in MW) with
  the case study's block bootstrap (14-day moving blocks over daily mean
  differences, 2000 resamples, seed 0), against a reference model;
- what a critical difference diagram would show: per-fold mean ranks, the
  Friedman test and the Nemenyi critical difference over the 8 folds.

The predictions are aligned on ``(prediction_time, split)`` across all the models
(17,466 rows), so every model is scored on the same rows. Tabular models use the
diff-feature runs (``cv_comparison_forecast_fg3``); Chronos-2 and t0-gamma use the
saved runs of ``cv_comparison_forecast``, which do not use the lagged features.

It needs a checkout of the case study (for ``uncertainty.py`` and
``compare_predictions.py``) and an environment with polars, numpy, scipy and
scikit-learn, for example the case study's own.

Usage::

    python scripts/paired_differences.py --case-study ../skore-example-electricity-load
"""

from __future__ import annotations

import argparse
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "raw_results"
MODELS = {
    "HGB quantile": "cv_comparison_forecast_fg3/hgb_quantile_q7_cv_predictions.parquet",
    "HGB + recal": "cv_comparison_forecast_fg3/hgb_quantile_recal_q7_cv_predictions.parquet",
    "HGB + conformal": "cv_comparison_forecast_fg3/hgb_quantile_conformal_q7_cv_predictions.parquet",
    "TabICL": "cv_comparison_forecast_fg3/tabicl_n8_full_cv_predictions.parquet",
    "Kumo-Tabular": "cv_comparison_forecast_fg3/kumo_tabular_cv_predictions.parquet",
    "Chronos-2": "cv_comparison_forecast/chronos2_ctx4000_cv_predictions.parquet",
    "t0-gamma": "cv_comparison_forecast/t0_gamma_cv_predictions.parquet",
}
Q_NEMENYI_005_K7 = 2.949  # Demsar (2006), table 5: Nemenyi q_0.05 for 7 models


def import_case_study(case_study: Path):
    """Import ``uncertainty`` and ``compare_predictions`` without the heavy package."""
    pkg = types.ModuleType("electricity_load_forecasting")
    pkg.__path__ = []
    sys.modules["electricity_load_forecasting"] = pkg
    spec = importlib.util.spec_from_file_location(
        "electricity_load_forecasting.metrics",
        case_study / "src/electricity_load_forecasting/metrics.py",
    )
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    pkg.metrics = metrics
    sys.path.insert(0, str(case_study / "scripts/model_comparison"))
    import compare_predictions
    import uncertainty

    return uncertainty, compare_predictions


def load_aligned(cp):
    frames = {name: cp.read_predictions(RAW / "model_comparison" / path) for name, path in MODELS.items()}
    keys = ["prediction_time", "split"]
    common = None
    for df in frames.values():
        part = df.select(keys).unique()
        common = part if common is None else common.join(part, on=keys, how="inner")
    out = {n: df.join(common, on=keys, how="inner").sort(["split", "prediction_time"]) for n, df in frames.items()}
    assert len({df.height for df in out.values()}) == 1
    return out


def pinball(U, df, horizon):
    return np.mean([U.row_loss(df, "pinball", horizon, q) for q in (0.05, 0.95)], axis=0)


def forest(U, frames, horizon, reference):
    """Pinball difference to ``reference`` (negative = better), with 95% interval."""
    ref = frames[reference]
    day, fold = U.row_index(ref)
    base = pinball(U, ref, horizon)
    print(f"\n== pinball at {horizon} h, difference vs {reference} (negative = better) ==")
    rows = []
    for name, df in frames.items():
        s = U.paired_difference(pinball(U, df, horizon), base, day, fold)
        rows.append((s["mean"], name, s))
    for _, name, s in sorted(rows, key=lambda r: r[0]):
        sig = "*" if not s["low"] < 0 < s["high"] else " "
        print(f"{sig} {name:16s} {s['mean']:+7.2f} [{s['low']:+7.2f}, {s['high']:+7.2f}]  lower in {s['n_lower']}/8 folds")


def ranks(U, frames, horizon):
    """Per-fold mean ranks, Friedman test and Nemenyi critical difference."""
    names = list(frames)
    k, n_folds = len(names), 8
    per_fold = []
    for name in names:
        df = frames[name]
        loss = pinball(U, df, horizon)
        fold = df["split"].to_numpy()
        per_fold.append([loss[fold == f].mean() for f in sorted(set(fold))])
    losses = np.array(per_fold)
    mean_rank = np.apply_along_axis(stats.rankdata, 0, losses).mean(axis=1)
    _, p = stats.friedmanchisquare(*losses)
    cd = Q_NEMENYI_005_K7 * np.sqrt(k * (k + 1) / (6 * n_folds))
    print(f"\n== mean ranks of the pinball loss at {horizon} h over {n_folds} folds ==")
    print(f"Friedman p = {p:.4f}, Nemenyi critical difference = {cd:.2f} ranks")
    for i in np.argsort(mean_rank):
        print(f"  {names[i]:16s} {mean_rank[i]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case-study", type=Path, default=REPO.parent / "skore-example-electricity-load")
    args = parser.parse_args()
    U, cp = import_case_study(args.case_study.resolve())
    frames = load_aligned(cp)
    print(f"aligned on {next(iter(frames.values())).height} rows")
    for horizon in (1, 24):
        forest(U, frames, horizon, "HGB quantile")
        ranks(U, frames, horizon)
    # Kumo-Tabular against the two corrected HGB variants, and the 1 h reference
    forest(U, frames, 24, "HGB + conformal")
    forest(U, frames, 24, "HGB + recal")
    forest(U, frames, 1, "Kumo-Tabular")


if __name__ == "__main__":
    main()
