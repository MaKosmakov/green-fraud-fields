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
    parser.add_argument("--out", default=None)
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
        "03b_permutation_placebo/permutation_placebo_summary.csv",
        "04_calibration_check/calibration_summary.csv",
        "04_calibration_check/top_tail_calibration_summary.csv",
        "05_posterior_uncertainty/mean_gains_vs_sd.csv",
        "06_final_bootstrap_ci/paired_block_bootstrap_ci.csv",
        "07_manuscript_consistency_audit/manuscript_number_audit.csv",
    ]
    records = [record for rel in key_files if (record := maybe_record(root, rel))]

    manifest = {
        "name": "Adaptive Green Risk Fields v2 block-causal frozen result manifest",
        "causal_policy": "strict_timestamp_block",
        "same_timestamp_policy": "score timestamp block before graph/history update",
        "label_release_policy": "release_time < candidate_timestamp",
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
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 40,
            },
            "tail_candidate_fractions": [0.025, 0.05, 0.10, 0.20, 0.30],
            "tail_selection_key": ["validation P@1%", "validation P@0.5%", "validation AUC-PR"],
            "logistic_tail": {"class_weight": "balanced", "C": 0.5, "max_iter": 2000},
            "crossfit_oof": {"folds": 3, "warmup_fraction": 0.5},
        },
        "diagnostics": {
            "bootstrap": {
                "replicates": 2000,
                "block_size_rows": 500,
                "interval": "percentile 2.5% and 97.5%",
            },
            "calibration": {"ece_bins": 10, "ece_binning": "equal-frequency"},
        },
        "result_files": records,
    }

    out = Path(args.out) if args.out else root / "frozen_result_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

