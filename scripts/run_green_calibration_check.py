from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import save_json


MODELS = {
    "M3": "score_M3",
    "M3_H_raw": "score_M3_H_raw",
    "M3_H_raw_S_D": "score_M3_H_raw_S_D",
    "green_tail_reranker": "score_adaptive_two_stage",
}


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def logit_clip(score: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(score.astype(float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def ece_equal_frequency(y: np.ndarray, score: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    order = np.argsort(score, kind="mergesort")
    chunks = np.array_split(order, bins)
    total = len(y)
    ece = 0.0
    rows = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        predicted = float(np.mean(score[chunk]))
        observed = float(np.mean(y[chunk]))
        weight = len(chunk) / total
        gap = abs(observed - predicted)
        ece += weight * gap
        rows.append(
            {
                "bin": idx,
                "n": int(len(chunk)),
                "score_min": float(np.min(score[chunk])),
                "score_max": float(np.max(score[chunk])),
                "mean_score": predicted,
                "observed_rate": observed,
                "abs_gap": gap,
            }
        )
    return float(ece), rows


def calibration_slope_intercept(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    x = logit_clip(score).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(x, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def top_tail_rows(y: np.ndarray, score: np.ndarray, budgets: tuple[float, ...]) -> list[dict]:
    order = np.argsort(-score, kind="mergesort")
    rows = []
    for budget in budgets:
        k = max(1, int(len(score) * budget))
        idx = order[:k]
        mean_score = float(np.mean(score[idx]))
        observed = float(np.mean(y[idx]))
        rows.append(
            {
                "budget": budget,
                "k": int(k),
                "mean_score_top": mean_score,
                "observed_rate_top": observed,
                "top_calibration_gap": observed - mean_score,
                "top_calibration_ratio": observed / mean_score if mean_score > 0 else float("nan"),
            }
        )
    return rows


def run_window(args: argparse.Namespace, window: int) -> tuple[list[dict], list[dict], list[dict]]:
    pred_path = Path(args.prediction_dir) / f"window_{window}" / "predictions_test.parquet"
    predictions = pd.read_parquet(pred_path)
    data = load_ieee_cis_cached(
        args.data_dir,
        (window + 1) * args.window_size,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    )
    data = data.iloc[window * args.window_size : (window + 1) * args.window_size].reset_index(drop=True)
    _, _, test = chronological_split(len(data))
    test_data = data.iloc[test].reset_index(drop=True)
    if not np.array_equal(predictions["TransactionID"].to_numpy(), test_data["TransactionID"].to_numpy()):
        raise ValueError(f"Prediction/test TransactionID mismatch for window {window}")
    y = test_data["isFraud"].to_numpy(int)

    metric_rows = []
    bin_rows = []
    tail_rows = []
    budgets = tuple(float(x) for x in args.tail_budgets.split(","))
    for model, column in MODELS.items():
        if column not in predictions.columns:
            raise ValueError(f"Missing {column} in {pred_path}")
        score = np.clip(predictions[column].to_numpy(float), 1e-6, 1 - 1e-6)
        ece, bins = ece_equal_frequency(y, score, args.ece_bins)
        slope, intercept = calibration_slope_intercept(y, score)
        metric_rows.append(
            {
                "window": window,
                "model": model,
                "n": int(len(y)),
                "prevalence": float(np.mean(y)),
                "mean_score": float(np.mean(score)),
                "brier": float(brier_score_loss(y, score)),
                "ece_equal_freq": ece,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
        for row in bins:
            bin_rows.append({"window": window, "model": model, **row})
        for row in top_tail_rows(y, score, budgets):
            tail_rows.append({"window": window, "model": model, **row})
    return metric_rows, bin_rows, tail_rows


def aggregate_metrics(metrics: pd.DataFrame, tails: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "prevalence",
        "mean_score",
        "brier",
        "ece_equal_freq",
        "calibration_slope",
        "calibration_intercept",
    ]
    metric_summary = metrics.groupby("model", as_index=False)[metric_cols].mean()
    metric_summary["windows"] = metrics.groupby("model")["window"].nunique().to_numpy()
    tail_summary = (
        tails.groupby(["model", "budget"], as_index=False)[
            ["mean_score_top", "observed_rate_top", "top_calibration_gap", "top_calibration_ratio"]
        ]
        .mean()
        .sort_values(["budget", "model"])
    )
    return metric_summary.sort_values("brier"), tail_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--prediction-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/01_graph_vs_history")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/04_calibration_check")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--tail-budgets", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--force-data-cache", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_metrics, all_bins, all_tails = [], [], []
    for window in parse_ints(args.windows):
        metrics, bins, tails = run_window(args, window)
        all_metrics.extend(metrics)
        all_bins.extend(bins)
        all_tails.extend(tails)

    metrics_frame = pd.DataFrame(all_metrics).sort_values(["window", "model"])
    bins_frame = pd.DataFrame(all_bins).sort_values(["window", "model", "bin"])
    tails_frame = pd.DataFrame(all_tails).sort_values(["window", "model", "budget"])
    metric_summary, tail_summary = aggregate_metrics(metrics_frame, tails_frame)

    metrics_frame.to_csv(out / "per_window_calibration.csv", index=False)
    bins_frame.to_csv(out / "calibration_bins.csv", index=False)
    tails_frame.to_csv(out / "top_tail_calibration.csv", index=False)
    metric_summary.to_csv(out / "calibration_summary.csv", index=False)
    tail_summary.to_csv(out / "top_tail_calibration_summary.csv", index=False)
    payload = {
        "models": list(MODELS),
        "windows": list(parse_ints(args.windows)),
        "ece_bins": args.ece_bins,
        "tail_budgets": [float(x) for x in args.tail_budgets.split(",")],
        "test_tuning": False,
        "changes_headline_ranking": False,
        "note": "Raw-score calibration diagnostics only; no recalibration or model selection was performed.",
    }
    save_json(out / "calibration_check.json", payload)
    report = [
        "# Calibration check",
        "",
        "Raw-score calibration diagnostics for the v2 block-causal Run 1 predictions.",
        "No recalibration, no test tuning, and no headline-ranking changes were performed.",
        "",
        "## Mean calibration metrics",
        "",
        metric_summary.to_csv(index=False),
        "",
        "## Mean top-tail calibration",
        "",
        tail_summary.to_csv(index=False),
    ]
    (out / "calibration_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"summary": metric_summary.to_dict("records")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
