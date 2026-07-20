from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import evaluate, save_json
from run_green_review_delay_sweep import write_aggregate_outputs as write_delay_aggregates
from run_green_review_graph_history import (
    METRICS as GRAPH_METRICS,
    RUN_SUBDIR as GRAPH_SUBDIR,
    exposure_counts,
    summarize_strata,
    write_aggregate_outputs as write_graph_aggregates,
)
from run_green_risk_field import cohort_metrics, green_columns


EVALUATION_POLICY = {
    "alert_tie_policy": "stable_arrival_order",
    "sort_algorithm": "numpy_mergesort",
    "test_labels_used_for_selection": False,
    "scores_refit": False,
}


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def score_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column.removeprefix("score_"): frame[column].to_numpy(float)
        for column in frame.columns
        if column.startswith("score_")
        and not column.endswith("_online")
        and not column.endswith("_batch_diagnostic")
    }


def canonicalize_pointwise_tail_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Promote frozen pointwise scores and retain old batch scores as diagnostics."""
    frame = frame.copy()
    for canonical in ("adaptive_two_stage", "crossfit_logistic_tail"):
        current = f"score_{canonical}"
        online = f"{current}_online"
        diagnostic = f"{current}_batch_diagnostic"
        if online in frame:
            if current in frame and diagnostic not in frame:
                frame[diagnostic] = frame[current]
            frame[current] = frame[online]
            frame = frame.drop(columns=[online])
    return frame


def assert_ids(left: pd.DataFrame, right: pd.DataFrame, context: str) -> None:
    if not np.array_equal(left["TransactionID"].to_numpy(), right["TransactionID"].to_numpy()):
        raise ValueError(f"TransactionID mismatch: {context}")


def update_test_metrics(path: Path, y: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updated = {}
    for model, score in predictions.items():
        validation = old.get(model, {}).get("validation", {"not_recomputed": True})
        updated[model] = {"validation": validation, "test": evaluate(y, score)}
    updated["_evaluation_policy"] = EVALUATION_POLICY
    save_json(path, updated)
    return updated


def rebuild_graph(args: argparse.Namespace, windows: tuple[int, ...]) -> None:
    root = Path(args.root)
    graph_root = root / GRAPH_SUBDIR
    headline_tail_root = root / args.headline_tail_subdir
    baseline_root = root / args.baseline_subdir
    adaptive_root = root / args.adaptive_subdir
    for window in windows:
        out = graph_root / f"window_{window}"
        graph_predictions = canonicalize_pointwise_tail_columns(pd.read_parquet(out / "predictions_test.parquet"))
        headline_predictions = canonicalize_pointwise_tail_columns(
            pd.read_parquet(headline_tail_root / f"window_{window}" / "predictions_test.parquet")
        )
        assert_ids(graph_predictions, headline_predictions, f"graph/headline-tail window {window}")
        for column in (
            "score_M3",
            "score_M3_H_raw",
            "score_M3_H_raw_S_D",
            "score_adaptive_soft",
            "score_adaptive_two_stage",
            "score_crossfit_logistic_tail",
        ):
            graph_predictions[column] = headline_predictions[column].to_numpy(float)
        y = headline_predictions["isFraud"].to_numpy(int)
        predictions = score_columns(graph_predictions)
        metrics = update_test_metrics(out / "metrics.json", y, predictions)

        base = pd.read_parquet(baseline_root / f"window_{window}" / "base_features.parquet").reset_index(drop=True)
        _, _, test = chronological_split(len(base))
        base_test = base.iloc[test].reset_index(drop=True)
        cohorts = cohort_metrics(base_test, y, predictions)
        save_json(out / "cohort_metrics.json", {"rows": cohorts, "evaluation_policy": EVALUATION_POLICY})

        green_path = baseline_root / f"window_{window}" / "green_features.parquet"
        schema = pq.ParquetFile(green_path).schema.names
        hcols = green_columns(schema, args.delay, args.alpha, "H")
        green = pd.read_parquet(green_path, columns=hcols).reset_index(drop=True)
        exposure_test = exposure_counts(green, hcols).iloc[test].reset_index(drop=True)
        strata = summarize_strata(window, base_test, exposure_test, y, predictions)
        pd.DataFrame(strata).to_csv(out / "count_strata_metrics.csv", index=False)

        rows = []
        runtime_path = out / "runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
        for model, payload in metrics.items():
            if model.startswith("_"):
                continue
            rows.append(
                {
                    "window": window,
                    "model": model,
                    "runtime_seconds": runtime.get("seconds", np.nan),
                    **{metric: payload["test"].get(metric, np.nan) for metric in GRAPH_METRICS},
                }
            )
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        graph_predictions.to_parquet(out / "predictions_test.parquet", index=False)
        save_json(out / "evaluation_policy.json", EVALUATION_POLICY)
        print(f"rebuilt graph metrics: window {window}", flush=True)
    write_graph_aggregates(root)


def rebuild_crossfit(args: argparse.Namespace, windows: tuple[int, ...]) -> None:
    root = Path(args.root) / args.crossfit_subdir
    baseline_root = Path(args.root) / args.baseline_subdir
    for window in windows:
        out = root / f"window_{window}"
        frame = pd.read_parquet(out / "predictions_test.parquet")
        y = frame["isFraud"].to_numpy(int)
        predictions = score_columns(frame)
        metrics = update_test_metrics(out / "metrics.json", y, predictions)
        base = pd.read_parquet(baseline_root / f"window_{window}" / "base_features.parquet").reset_index(drop=True)
        _, _, test = chronological_split(len(base))
        cohorts = cohort_metrics(base.iloc[test].reset_index(drop=True), y, predictions)
        save_json(out / "cohort_metrics.json", {"rows": cohorts, "evaluation_policy": EVALUATION_POLICY})
        runtime_path = out / "runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
        rows = []
        for model, payload in metrics.items():
            if model.startswith("_"):
                continue
            rows.append(
                {
                    "window": window,
                    "model": model,
                    "auc_pr": payload["test"]["auc_pr"],
                    "precision_at_0.005": payload["test"]["precision_at_0.005"],
                    "precision_at_0.01": payload["test"]["precision_at_0.01"],
                    "precision_at_0.02": payload["test"]["precision_at_0.02"],
                    "precision_at_0.05": payload["test"]["precision_at_0.05"],
                    "runtime_seconds": runtime.get("seconds", np.nan),
                }
            )
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        save_json(out / "evaluation_policy.json", EVALUATION_POLICY)
        print(f"rebuilt reranker metrics: window {window}", flush=True)


def rebuild_delay(args: argparse.Namespace, windows: tuple[int, ...], delays: tuple[int, ...]) -> None:
    root = Path(args.root) / args.delay_subdir
    baseline_root = Path(args.root) / args.baseline_subdir
    for delay in delays:
        for window in windows:
            out = root / f"delay_{delay}" / f"window_{window}"
            frame = canonicalize_pointwise_tail_columns(pd.read_parquet(out / "predictions_test.parquet"))
            y = frame["isFraud"].to_numpy(int)
            predictions = score_columns(frame)
            metrics = update_test_metrics(out / "metrics.json", y, predictions)
            base = pd.read_parquet(baseline_root / f"window_{window}" / "base_features.parquet").reset_index(drop=True)
            _, _, test = chronological_split(len(base))
            cohorts = cohort_metrics(base.iloc[test].reset_index(drop=True), y, predictions)
            save_json(out / "cohort_metrics.json", {"rows": cohorts, "evaluation_policy": EVALUATION_POLICY})
            runtime_path = out / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
            rows = []
            for model, payload in metrics.items():
                if model.startswith("_"):
                    continue
                rows.append(
                    {
                        "delay": delay,
                        "window": window,
                        "model": model,
                        "runtime_seconds": runtime.get("seconds", np.nan),
                        **{metric: payload["test"].get(metric, np.nan) for metric in GRAPH_METRICS},
                    }
                )
            pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
            frame.to_parquet(out / "predictions_test.parquet", index=False)
            save_json(out / "evaluation_policy.json", EVALUATION_POLICY)
            print(f"rebuilt delay metrics: delay {delay}, window {window}", flush=True)
    write_delay_aggregates(Path(args.root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate frozen test predictions with deterministic alert ties.")
    parser.add_argument("--root", default="outputs/ieee_green_final_review_gates_v2_block_causal")
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--baseline-subdir", default="00_moderate_100k_block_causal")
    parser.add_argument("--adaptive-subdir", default="00_adaptive_theory_block_causal")
    parser.add_argument("--crossfit-subdir", default="00_crossfit_reranker_block_causal")
    parser.add_argument("--headline-tail-subdir", default="02_delay_sweep/delay_0")
    parser.add_argument("--delay-subdir", default="02_delay_sweep")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--scope", choices=("all", "graph", "crossfit", "delay"), default="all")
    args = parser.parse_args()
    windows = parse_ints(args.windows)
    delays = parse_ints(args.delays)
    if args.scope in {"all", "crossfit"}:
        rebuild_crossfit(args, windows)
    if args.scope in {"all", "graph"}:
        rebuild_graph(args, windows)
    if args.scope in {"all", "delay"}:
        rebuild_delay(args, windows, delays)


if __name__ == "__main__":
    main()
