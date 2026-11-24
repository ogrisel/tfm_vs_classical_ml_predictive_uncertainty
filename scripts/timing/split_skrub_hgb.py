"""
Split the fit / predict time of the HGB pipelines between the skrub DataOps graph
(loading, feature engineering, conversions) and the scikit-learn HGB estimators.

The case study's CV scores only give the total fit and score time of each fold.
Here, one fold is fitted and scored with the same learner as
`electricity_load_forecasting.cross_val_predict`, while every call to
`BaseHistGradientBoosting.fit` / `.predict` is timed. The skrub part is the
remainder: total - HGB.

`max_train_rows` (the cap applied to the training rows inside the graph, in fit
mode only) gives the training-size subsets. Scoring always uses the full test fold.

Run from this repository's root, with the case study source on the path:

    PYTHONPATH=/path/to/skore-example-electricity-load/src \
        python scripts/timing/split_skrub_hgb.py --model hgb_mean \
        --max_train_rows 1000 2000 4000 0 --folds 0 7

`--max_train_rows 0` means no cap (the full training fold, as in the CV runs).
Results are appended to `raw_results/timing/split_<model>.csv`.
"""

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import electricity_load_forecasting as elf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    BaseHistGradientBoosting,
)

FEATURE_GROUPS = ("load_diffs", "load_same_hour", "temp_diffs")
HORIZONS = (1, 12, 24)
QUANTILES_7 = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

STATS = {"fit": 0.0, "predict": 0.0, "fit_calls": 0, "predict_calls": 0,
         "fit_rows": 0, "predict_rows": 0, "depth": 0}

_orig_fit = BaseHistGradientBoosting.fit
_orig_predict = HistGradientBoostingRegressor.predict


def _timed_fit(self, X, y, *args, **kwargs):
    if STATS["depth"]:
        return _orig_fit(self, X, y, *args, **kwargs)
    STATS["depth"] += 1
    t0 = time.perf_counter()
    try:
        return _orig_fit(self, X, y, *args, **kwargs)
    finally:
        STATS["fit"] += time.perf_counter() - t0
        STATS["fit_calls"] += 1
        STATS["fit_rows"] += len(X)
        STATS["depth"] -= 1


def _timed_predict(self, X, *args, **kwargs):
    if STATS["depth"]:
        return _orig_predict(self, X, *args, **kwargs)
    STATS["depth"] += 1
    t0 = time.perf_counter()
    try:
        return _orig_predict(self, X, *args, **kwargs)
    finally:
        STATS["predict"] += time.perf_counter() - t0
        STATS["predict_calls"] += 1
        STATS["predict_rows"] += len(X)
        STATS["depth"] -= 1


BaseHistGradientBoosting.fit = _timed_fit
HistGradientBoostingRegressor.predict = _timed_predict


def make_pred(model, max_train_rows):
    kwargs = dict(
        horizons=HORIZONS,
        feature_groups=FEATURE_GROUPS,
        max_train_rows=max_train_rows or None,
    )
    if model == "hgb_mean":
        return elf.make_data_op(model="hgb_mean", max_iter=3000, **kwargs)
    if model == "hgb_quantile":
        return elf.make_data_op(
            model="hgb_quantile", quantiles=QUANTILES_7, max_iter=3000, **kwargs
        )
    raise ValueError(model)


def reset_stats():
    for k in STATS:
        STATS[k] = 0 if isinstance(STATS[k], int) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hgb_mean", "hgb_quantile"], required=True)
    parser.add_argument("--max_train_rows", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default=Path("raw_results/timing"))
    args = parser.parse_args()

    env = elf.get_env() | {"city_weather_fetcher": elf.fetch_city_weather_forecast}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"split_{args.model}.csv"
    fields = [
        "model", "fold", "max_train_rows", "rep", "fold_train_rows", "fit_rows_hgb",
        "fit_calls", "fit_total_s", "fit_hgb_s", "fit_skrub_s", "test_rows",
        "predict_rows_hgb", "predict_calls", "predict_total_s", "predict_hgb_s",
        "predict_skrub_s", "cpu_count", "machine", "omp_threads",
    ]
    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for cap in args.max_train_rows:
            pred = make_pred(args.model, cap)
            splits = list(pred.skb.iter_cv_splits(environment=env))
            for fold in args.folds:
                split = splits[fold]
                for rep in range(args.reps):
                    reset_stats()
                    t0 = time.perf_counter()
                    learner = pred.skb.make_learner().fit(split["train"])
                    fit_total = time.perf_counter() - t0
                    fit_stats = dict(STATS)
                    reset_stats()
                    t0 = time.perf_counter()
                    learner.score(split["test"], return_predictions=True)
                    score_total = time.perf_counter() - t0
                    pred_stats = dict(STATS)
                    row = {
                        "model": args.model,
                        "fold": fold,
                        "max_train_rows": cap,
                        "rep": rep,
                        "fold_train_rows": split["X_train"].height,
                        "fit_rows_hgb": fit_stats["fit_rows"],
                        "fit_calls": fit_stats["fit_calls"],
                        "fit_total_s": round(fit_total, 3),
                        "fit_hgb_s": round(fit_stats["fit"], 3),
                        "fit_skrub_s": round(fit_total - fit_stats["fit"], 3),
                        "test_rows": split["X_test"].height,
                        "predict_rows_hgb": pred_stats["predict_rows"],
                        "predict_calls": pred_stats["predict_calls"],
                        "predict_total_s": round(score_total, 3),
                        "predict_hgb_s": round(pred_stats["predict"], 3),
                        "predict_skrub_s": round(score_total - pred_stats["predict"], 3),
                        "cpu_count": os.cpu_count(),
                        "machine": platform.machine() + " " + platform.processor(),
                        "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
                    }
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"{args.model} cap={cap or 'all'} fold={fold} rep={rep}: "
                        f"fit {fit_total:.1f}s (hgb {fit_stats['fit']:.1f}s, "
                        f"{fit_stats['fit_rows']} rows, {fit_stats['fit_calls']} fits), "
                        f"predict {score_total:.1f}s (hgb {pred_stats['predict']:.1f}s)",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
