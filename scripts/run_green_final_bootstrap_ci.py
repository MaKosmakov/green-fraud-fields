from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import save_json


COMPARISONS = {
    "graph_marginal": ("score_M3_H_raw", "score_M3_H_raw_S_D"),
    "tail_vs_m3": ("score_M3", "score_adaptive_two_stage"),
    "crossfit_vs_m3": ("score_M3", "score_crossfit_logistic_tail"),
    "tail_vs_history": ("score_M3_H_raw", "score_adaptive_two_stage"),
    "crossfit_vs_history": ("score_M3_H_raw", "score_crossfit_logistic_tail"),
}

BUDGETS = (0.005, 0.01, 0.02, 0.05)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def precision_at(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    k = max(1, int(len(score) * budget))
    idx = np.argsort(-score)[:k]
    return float(np.mean(y[idx]))


def metric_delta(y: np.ndarray, base: np.ndarray, alt: np.ndarray) -> dict[str, float]:
    output = {
        "auc_pr": float(average_precision_score(y, alt) - average_precision_score(y, base)),
    }
    for budget in BUDGETS:
        output[f"precision_at_{budget:g}"] = precision_at(y, alt, budget) - precision_at(y, base, budget)
    return output


def bootstrap_indices_by_window(
    windows: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    pieces = []
    for window in sorted(np.unique(windows)):
        window_positions = np.flatnonzero(windows == window)
        unique_blocks = np.unique(block_ids[window_positions])
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        for block in sampled_blocks:
            pieces.append(window_positions[block_ids[window_positions] == block])
    return np.concatenate(pieces)


def paired_block_bootstrap(
    frame: pd.DataFrame,
    comparison: str,
    mask: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    base_col, alt_col = COMPARISONS[comparison]
    subset = frame.loc[mask].reset_index(drop=True)
    y = subset["isFraud"].to_numpy(int)
    base = subset[base_col].to_numpy(float)
    alt = subset[alt_col].to_numpy(float)
    if len(y) < 100 or np.unique(y).size < 2:
        return {
            "comparison": comparison,
            "rows": int(len(y)),
            "frauds": int(y.sum()),
            "skipped": True,
            "reason": "too few rows or only one class",
        }
    observed = metric_delta(y, base, alt)
    windows = subset["window"].to_numpy(int)
    block_ids = subset["block_id"].to_numpy(int)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        idx = bootstrap_indices_by_window(windows, block_ids, rng)
        if np.unique(y[idx]).size < 2:
            continue
        draws.append(metric_delta(y[idx], base[idx], alt[idx]))
    draw_frame = pd.DataFrame(draws)
    row = {
        "comparison": comparison,
        "rows": int(len(y)),
        "frauds": int(y.sum()),
        "replicates": int(len(draw_frame)),
        "skipped": False,
    }
    for metric in observed:
        values = draw_frame[metric].to_numpy(float)
        row[f"observed_{metric}_delta"] = observed[metric]
        row[f"bootstrap_mean_{metric}_delta"] = float(np.mean(values))
        row[f"ci_low_{metric}_delta"] = float(np.quantile(values, 0.025))
        row[f"ci_high_{metric}_delta"] = float(np.quantile(values, 0.975))
        row[f"p_gt_0_{metric}_delta"] = float(np.mean(values > 0))
    return row


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for window in parse_ints(args.windows):
        pred_path = Path(args.prediction_dir) / f"window_{window}" / "predictions_test.parquet"
        base_path = Path(args.base_dir) / f"window_{window}" / "base_features.parquet"
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
        base_test = pd.read_parquet(base_path).iloc[test].reset_index(drop=True)
        if not np.array_equal(predictions["TransactionID"].to_numpy(), test_data["TransactionID"].to_numpy()):
            raise ValueError(f"TransactionID mismatch in window {window}")
        frame = predictions.copy()
        frame["isFraud"] = test_data["isFraud"].to_numpy(int)
        frame["window"] = window
        frame["position_in_test"] = np.arange(len(frame))
        frame["block_id"] = frame["position_in_test"] // args.block_size
        frame["cohort_all"] = True
        frame["cohort_known_endpoints"] = base_test["known_endpoints_all_edges"].to_numpy(bool)
        frame["cohort_C00_newedge"] = base_test["cohort_any_C00_newedge"].to_numpy(bool)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--prediction-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/01_graph_vs_history")
    parser.add_argument("--base-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/00_moderate_100k_block_causal")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/06_final_bootstrap_ci")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--force-data-cache", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args)
    frame.to_parquet(out / "bootstrap_input.parquet", index=False)
    rows = []
    cohorts = {
        "all": frame["cohort_all"].to_numpy(bool),
        "known_endpoints": frame["cohort_known_endpoints"].to_numpy(bool),
        "C00_newedge": frame["cohort_C00_newedge"].to_numpy(bool),
    }
    for cohort, mask in cohorts.items():
        for comparison in COMPARISONS:
            row = paired_block_bootstrap(frame, comparison, mask, args.replicates, args.seed + len(rows))
            row["cohort"] = cohort
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "paired_block_bootstrap_ci.csv", index=False)
    save_json(
        out / "paired_block_bootstrap_ci.json",
        {
            "replicates": args.replicates,
            "block_size": args.block_size,
            "windows": list(parse_ints(args.windows)),
            "comparisons": COMPARISONS,
            "budgets": BUDGETS,
            "rows": result.to_dict("records"),
        },
    )
    report = [
        "# Final paired block bootstrap confidence intervals",
        "",
        f"Replicates: {args.replicates}; chronological block size: {args.block_size}; windows: {args.windows}.",
        "",
        result.to_csv(index=False),
    ]
    (out / "paired_block_bootstrap_ci_report.md").write_text("\n".join(report), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
