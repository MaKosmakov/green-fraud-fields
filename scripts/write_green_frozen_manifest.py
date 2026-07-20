from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_record(root: Path, relative: str) -> dict | None:
    path = root / relative
    if not path.exists():
        return None
    return {
        "relative_to_output_root": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="outputs/ieee_green_final_review_gates_v2_block_causal",
        help="Root of the v2 block-causal result directory.",
    )
    parser.add_argument("--out", default="manifests/ieee_green_block_causal_manifest.json")
    args = parser.parse_args()

    root = Path(args.output_root)
    key_files = [
        "01_graph_vs_history/window_summary.csv",
        "01_graph_vs_history/mean_gains_vs_m3.csv",
        "01_graph_vs_history/mean_gains_vs_history.csv",
        "01_graph_vs_history/graph_marginal_summary.csv",
        "01_graph_vs_history/count_strata_mean_gains.csv",
        "01_graph_vs_history/critical_cohort_graph_margins.csv",
        "02_delay_sweep/per_delay_window_summary.csv",
        "02_delay_sweep/per_delay_mean_gains_vs_m3.csv",
        "02_delay_sweep/per_delay_mean_gains_vs_history.csv",
        "02_delay_sweep/delay_graph_marginal_summary.csv",
        "02_delay_sweep/critical_cohort_delay_gains.csv",
        "03_leakage_audit_final/leakage_audit_results.json",
        "03_leakage_audit_final/cache_provenance.csv",
        "03_leakage_audit_final/selection_audit.csv",
        "03b_permutation_placebo/permutation_placebo_summary.csv",
        "04_calibration_check/calibration_summary.csv",
        "04_calibration_check/top_tail_calibration_summary.csv",
        "08_online_causal_audit/paired_block_bootstrap_ci.csv",
    ]
    records = [record for rel in key_files if (record := maybe_record(root, rel))]

    manifest = {
        "name": "Adaptive Green Risk Fields block-causal final result manifest",
        "causal_policy": "strict_timestamp_block",
        "same_timestamp_policy": "score timestamp block before graph/history update",
        "label_release_policy": "release_time < candidate_timestamp",
        "window_state_policy": "reset graph and released history at each 100k headline window",
        "alert_tie_policy": "stable chronological arrival order",
        "windows": [0, 1, 2, 3, 4],
        "split_rows": {"train": 60000, "validation": 20000, "test": 20000},
        "green_features": {
            "risk_layers": [
                "card1--addr1",
                "card1--P_emaildomain",
                "card1--DeviceInfo",
                "DeviceInfo--P_emaildomain",
            ],
            "edge_weight": "unit historical co-occurrence count",
            "ego_radius": 2,
            "max_ego_nodes": 100,
            "plain_green_lambda": 0.1,
            "history_alpha": 5,
            "adaptive_confidence": "d_v = 5 + sqrt(n_v)",
            "delays_days": [0, 1, 3, 7, 14],
        },
        "modeling": {
            "random_forest": "not used",
            "lightgbm": {
                "n_estimators_max": 700,
                "learning_rate": 0.04,
                "candidate_grid": [
                    {"num_leaves": 15, "min_child_samples": 40, "reg_lambda": 2.0},
                    {"num_leaves": 31, "min_child_samples": 20, "reg_lambda": 1.0},
                    {"num_leaves": 63, "min_child_samples": 30, "reg_lambda": 3.0},
                    {"num_leaves": 31, "min_child_samples": 80, "reg_lambda": 5.0},
                ],
                "subsample": 0.9,
                "subsample_freq": 0,
                "row_bagging_active": False,
                "colsample_bytree": 0.9,
                "early_stopping_rounds": 50,
                "eval_metric": "average_precision",
                "selection_key": ["validation P@1%", "validation P@0.5%", "validation P@2%", "validation AUC-PR"],
                "scale_pos_weight": "computed from training prevalence",
            },
            "tail_candidate_fractions": [0.025, 0.05, 0.10, 0.20, 0.30],
            "tail_selection_key": ["validation P@1%", "validation P@0.5%", "validation AUC-PR"],
            "logistic_tail": {"class_weight": "balanced", "C": 0.5, "max_iter": 2000},
            "online_tail_score": {
                "cutoff_source": "validation only",
                "noncandidate": "0.5 * base_probability",
                "candidate": "0.5 + 0.5 * second_stage_probability",
                "prefix_invariant": True,
                "test_batch_quantile_or_rank": False,
            },
            "crossfit_oof": {
                "folds": 3,
                "warmup_fraction": 0.5,
                "n_estimators_max": 160,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 40,
                "reg_lambda": 3.0,
                "early_stopping_rounds": 25,
            },
        },
        "diagnostics": {
            "bootstrap": {
                "replicates": 2000,
                "block_size_rows": 500,
                "interval": "percentile 2.5% and 97.5%",
            },
            "calibration": {"ece_bins": 10, "ece_binning": "equal-frequency"},
        },
        "source_scripts": [
            "scripts/build_green_block_causal_features.py",
            "scripts/build_adaptive_precision_block_causal.py",
            "scripts/run_green_review_graph_history.py",
            "scripts/run_green_review_delay_sweep.py",
            "scripts/rebuild_saved_prediction_metrics.py",
            "scripts/run_green_leakage_audit.py",
            "scripts/run_green_permutation_placebo.py",
            "scripts/run_green_calibration_check.py",
            "scripts/run_green_final_bootstrap_ci.py",
            "scripts/verify_green_frozen_manifest.py",
            "scripts/plot_delay_auc_sweep.py"
        ],
        "result_files": records,
    }

    out = Path(args.out) if args.out else root / "frozen_result_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

