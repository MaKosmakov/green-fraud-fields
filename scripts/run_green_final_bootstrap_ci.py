from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "tail_online_vs_m3": ("score_M3", "score_adaptive_two_stage"),
    "crossfit_online_vs_m3": ("score_M3", "score_crossfit_logistic_tail"),
    "tail_online_vs_history": ("score_M3_H_raw", "score_adaptive_two_stage"),
    "crossfit_online_vs_history": ("score_M3_H_raw", "score_crossfit_logistic_tail"),
}

LEGACY_BATCH_COMPARISONS = {
    "tail_vs_m3": ("score_M3", "score_adaptive_two_stage_batch_diagnostic"),
    "crossfit_vs_m3": ("score_M3", "score_crossfit_logistic_tail_batch_diagnostic"),
    "tail_vs_history": ("score_M3_H_raw", "score_adaptive_two_stage_batch_diagnostic"),
    "crossfit_vs_history": ("score_M3_H_raw", "score_crossfit_logistic_tail_batch_diagnostic"),
}

BUDGETS = (0.005, 0.01, 0.02, 0.05)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def precision_at(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    k = max(1, int(len(score) * budget))
    idx = np.argsort(-score, kind="mergesort")[:k]
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


def prepare_weighted_ranking(
    y: np.ndarray,
    score: np.ndarray,
    row_block_codes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pre-sort a score once for integer-weighted block resamples."""
    order = np.argsort(-score, kind="mergesort")
    sorted_scores = score[order]
    tie_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(order) - 1]
    return {
        "y": y[order],
        "block": row_block_codes[order],
        "tie_ends": tie_ends,
    }


def weighted_rank_metrics(
    prepared: dict[str, np.ndarray],
    block_counts: np.ndarray,
) -> dict[str, float]:
    """Evaluate AP and alert precision for a duplicated-block resample.

    Integer block multiplicities are equivalent to materializing each sampled
    block that many times under the frozen arrival-order tie policy, while
    avoiding a fresh score sort per draw.
    """
    y = prepared["y"]
    weights = block_counts[prepared["block"]]
    cumulative_rows = np.cumsum(weights, dtype=np.int64)
    cumulative_frauds = np.cumsum(weights * y, dtype=np.int64)
    total_rows = int(cumulative_rows[-1])
    total_frauds = int(cumulative_frauds[-1])
    if total_frauds == 0:
        raise ValueError("Bootstrap draw contains no positive labels")

    ends = prepared["tie_ends"]
    rows_at_threshold = cumulative_rows[ends]
    frauds_at_threshold = cumulative_frauds[ends]
    frauds_by_threshold = np.diff(np.r_[0, frauds_at_threshold])
    valid = rows_at_threshold > 0
    average_precision = float(
        np.sum(
            (frauds_by_threshold[valid] / total_frauds)
            * (frauds_at_threshold[valid] / rows_at_threshold[valid])
        )
    )

    output = {"auc_pr": average_precision}
    for budget in BUDGETS:
        k = max(1, int(total_rows * budget))
        position = int(np.searchsorted(cumulative_rows, k, side="left"))
        rows_before = int(cumulative_rows[position - 1]) if position else 0
        frauds_before = int(cumulative_frauds[position - 1]) if position else 0
        take = k - rows_before
        frauds_in_top_k = frauds_before + take * int(y[position])
        output[f"precision_at_{budget:g}"] = float(frauds_in_top_k / k)
    return output


def sampled_block_counts(
    window_block_codes: list[np.ndarray],
    number_of_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.zeros(number_of_blocks, dtype=np.int32)
    for codes in window_block_codes:
        sampled = rng.choice(codes, size=len(codes), replace=True)
        counts += np.bincount(sampled, minlength=number_of_blocks).astype(np.int32)
    return counts


def paired_block_bootstrap(
    frame: pd.DataFrame,
    comparison: str,
    comparison_columns: tuple[str, str],
    mask: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    base_col, alt_col = comparison_columns
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
    block_keys = pd.MultiIndex.from_arrays([windows, block_ids])
    row_block_codes, unique_block_keys = pd.factorize(block_keys, sort=True)
    window_block_codes = []
    for window in sorted(np.unique(windows)):
        window_block_codes.append(np.unique(row_block_codes[windows == window]))
    base_prepared = prepare_weighted_ranking(y, base, row_block_codes)
    alt_prepared = prepare_weighted_ranking(y, alt, row_block_codes)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        counts = sampled_block_counts(window_block_codes, len(unique_block_keys), rng)
        try:
            base_metrics = weighted_rank_metrics(base_prepared, counts)
            alt_metrics = weighted_rank_metrics(alt_prepared, counts)
        except ValueError:
            continue
        draws.append({metric: alt_metrics[metric] - base_metrics[metric] for metric in alt_metrics})
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


def run_bootstrap_job(
    frame: pd.DataFrame,
    comparison: str,
    comparison_columns: tuple[str, str],
    cohort: str,
    replicates: int,
    seed: int,
) -> dict:
    """Run one cohort/comparison cell in a worker process."""
    row = paired_block_bootstrap(
        frame,
        comparison,
        comparison_columns,
        np.ones(len(frame), dtype=bool),
        replicates,
        seed,
    )
    row["cohort"] = cohort
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
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/08_online_causal_audit")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--include-legacy-batch",
        action="store_true",
        help="Also bootstrap the pre-audit test-batch-ranked tail scores.",
    )
    parser.add_argument("--force-data-cache", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args)
    frame.to_parquet(out / "bootstrap_input.parquet", index=False)
    comparisons = dict(COMPARISONS)
    if args.include_legacy_batch:
        comparisons.update(LEGACY_BATCH_COMPARISONS)
    rows = []
    cohorts = {
        "all": frame["cohort_all"].to_numpy(bool),
        "known_endpoints": frame["cohort_known_endpoints"].to_numpy(bool),
        "C00_newedge": frame["cohort_C00_newedge"].to_numpy(bool),
    }
    jobs = []
    job_number = 0
    for cohort, mask in cohorts.items():
        for comparison, columns in comparisons.items():
            needed = ["isFraud", "window", "block_id", *columns]
            job_frame = frame.loc[mask, needed].reset_index(drop=True)
            jobs.append((job_frame, comparison, columns, cohort, args.replicates, args.seed + job_number))
            job_number += 1

    checkpoint = out / "paired_block_bootstrap_ci.partial.csv"
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(run_bootstrap_job, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(
                f"completed {completed}/{len(futures)}: {row['cohort']} / {row['comparison']}",
                flush=True,
            )
    result = pd.DataFrame(rows).sort_values(["cohort", "comparison"]).reset_index(drop=True)
    result.to_csv(out / "paired_block_bootstrap_ci.csv", index=False)
    save_json(
        out / "paired_block_bootstrap_ci.json",
        {
            "replicates": args.replicates,
            "block_size": args.block_size,
            "windows": list(parse_ints(args.windows)),
            "comparisons": comparisons,
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
