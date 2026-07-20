from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from green_fraud_fields.green_risk_field import GreenRiskFieldBuilder
from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import evaluate, save_json
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached
from run_green_adaptive_theory import PrecisionWeightedGreenBuilder
from run_green_crossfit_reranker import crossfit_logistic_tail, oof_base_predictions
from run_green_focused_improvements import tuned_soft_mixture
from run_green_review_delay_sweep import parse_ints, summary_columns
from run_green_reranker_upgrade import logistic_tail_reranker
from run_green_risk_field import base_groups, cohort_metrics, green_columns


RUN_SUBDIR = "03b_permutation_placebo"
METRICS = [
    "auc_pr",
    "roc_auc",
    "precision_at_0.005",
    "precision_at_0.01",
    "precision_at_0.02",
    "precision_at_0.05",
]


def has_block_causal_metadata(path: Path, delay: int | None = None) -> bool:
    if not path.exists():
        return False
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    ok = (
        metadata.get(b"causal_policy") == b"strict_timestamp_block"
        and metadata.get(b"same_timestamp_policy") == b"block_frozen_t_minus"
        and metadata.get(b"label_release_policy") == b"release_time_strictly_less_than_candidate_timestamp"
    )
    if delay is not None and metadata.get(b"delay_days") not in {None, str(delay).encode()}:
        ok = ok and metadata.get(b"delay_days") == str(delay).encode()
    return ok


def permute_history_labels(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute labels used by causal released-history features only.

    The true labels in the modeling/evaluation arrays are kept unchanged. This
    destroys label-history signal while preserving the exact transaction times,
    graph edges, and label prevalence inside the window.
    """
    rng = np.random.default_rng(seed)
    out = data.copy()
    out["isFraud"] = rng.permutation(out["isFraud"].to_numpy(int))
    return out


def ensure_placebo_features(
    args: argparse.Namespace,
    data_perm: pd.DataFrame,
    out: Path,
) -> tuple[Path, Path, dict, dict]:
    green_path = out / "green_features_permuted_labels.parquet"
    sd_path = out / "precision_weighted_green_permuted_labels.parquet"
    green_runtime: dict
    sd_runtime: dict

    if green_path.exists() and has_block_causal_metadata(green_path):
        green_runtime = {"cached": True, "path": str(green_path)}
    else:
        if green_path.exists():
            green_path.unlink()
        builder = GreenRiskFieldBuilder(
            delays_days=(args.delay,),
            alphas=(args.alpha,),
            lambda_=0.1,
            ego_radius=args.ego_radius,
            max_ego_nodes=args.max_ego_nodes,
            dense_threshold=150,
        )
        started = time.time()
        builder.write_parquet(data_perm, str(green_path), batch_size=args.batch_size)
        green_runtime = {
            "cached": False,
            "path": str(green_path),
            "feature_seconds": time.time() - started,
            **builder.runtime_summary(),
        }

    if sd_path.exists() and has_block_causal_metadata(sd_path, args.delay):
        sd_runtime = {"cached": True, "path": str(sd_path)}
    else:
        if sd_path.exists():
            sd_path.unlink()
        builder = PrecisionWeightedGreenBuilder(
            delay_days=args.delay,
            history_alpha=args.alpha,
            precision_alpha=args.precision_alpha,
            radius=args.ego_radius,
            cap=args.max_ego_nodes,
        )
        sd_runtime = builder.write_parquet(data_perm, sd_path, batch_size=args.batch_size)
        sd_runtime["cached"] = False
        sd_runtime["path"] = str(sd_path)
    return green_path, sd_path, green_runtime, sd_runtime


def run_window(args: argparse.Namespace, window: int) -> dict:
    timer = StageTimer()
    root = Path(args.out_dir)
    out = root / RUN_SUBDIR / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)

    data = load_ieee_cis_cached(
        args.data_dir,
        (window + 1) * args.window_size,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    )
    data = data.iloc[window * args.window_size : (window + 1) * args.window_size].reset_index(drop=True)
    data_perm = permute_history_labels(data, args.seed + 1009 * window)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    timer.mark("load_and_permute_data")

    baseline_window = Path(args.baseline_dir) / f"window_{window}"
    base = pd.read_parquet(baseline_window / "base_features.parquet").reset_index(drop=True)
    green_path, sd_path, green_runtime, sd_runtime = ensure_placebo_features(args, data_perm, out)
    timer.mark("compute_or_load_placebo_features")

    schema = pq.ParquetFile(green_path).schema.names
    hcols = green_columns(schema, args.delay, args.alpha, "H")
    if not hcols:
        raise ValueError(f"No placebo H columns found in {green_path}")
    green = pd.read_parquet(green_path, columns=hcols).reset_index(drop=True)
    sd_schema = pq.ParquetFile(sd_path).schema.names
    sd_cols = [c for c in sd_schema if "SD_sqrt" in c]
    if not sd_cols:
        raise ValueError(f"No placebo SD_sqrt columns found in {sd_path}")
    sd = pd.read_parquet(sd_path, columns=sd_cols).reset_index(drop=True)
    frame = pd.concat([base, green, sd], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    timer.mark("assemble_frame")

    baseline_cols = base_groups(list(base.columns))["BP"]
    model_cols = {
        "M3": baseline_cols,
        "M3_H_raw": baseline_cols + hcols,
        "M3_H_raw_S_D": baseline_cols + hcols + sd_cols,
    }
    valid_predictions, test_predictions, selection, model_timing = fit_named_models_cached(
        frame,
        y,
        train,
        valid,
        test,
        model_cols,
        args.seed,
        out / "model_prediction_cache",
        args.force_models,
    )
    timer.mark("fit_core_models")

    components = ["M3", "M3_H_raw", "M3_H_raw_S_D"]
    soft_valid, soft_test, soft_selection = tuned_soft_mixture(
        y[valid],
        valid_predictions,
        test_predictions,
        components,
    )
    valid_predictions["adaptive_soft"] = soft_valid
    test_predictions["adaptive_soft"] = soft_test
    selection["adaptive_soft"] = soft_selection
    timer.mark("adaptive_soft")

    meta_models = ["M3", "M3_H_raw", "M3_H_raw_S_D", "adaptive_soft"]
    meta_cols = summary_columns(hcols, sd_cols)
    two_valid, two_test, two_selection = logistic_tail_reranker(
        frame,
        y,
        valid,
        test,
        valid_predictions,
        test_predictions,
        meta_models,
        meta_cols,
        fractions=(0.025, 0.05, 0.10, 0.20, 0.30),
        online_causal=True,
    )
    valid_predictions["adaptive_two_stage"] = two_valid
    test_predictions["adaptive_two_stage"] = two_test
    selection["adaptive_two_stage"] = two_selection
    timer.mark("adaptive_two_stage")

    if args.include_crossfit:
        oof_index, oof_predictions, oof_details = oof_base_predictions(
            frame,
            y,
            train,
            model_cols,
            folds=args.oof_folds,
            warmup_fraction=args.oof_warmup_fraction,
            seed=args.seed,
            cache_dir=out / "oof_prediction_cache",
            force=args.force_oof,
        )
        cross_valid, cross_test, cross_selection = crossfit_logistic_tail(
            frame,
            y,
            oof_index,
            valid,
            test,
            oof_predictions,
            valid_predictions,
            test_predictions,
            list(model_cols),
            meta_cols,
            fractions=(0.025, 0.05, 0.10, 0.20, 0.30),
            online_causal=True,
        )
        valid_predictions["crossfit_logistic_tail"] = cross_valid
        test_predictions["crossfit_logistic_tail"] = cross_test
        selection["crossfit_logistic_tail"] = cross_selection
        selection["oof_base_predictions"] = oof_details
    timer.mark("crossfit_optional")

    metrics = {
        name: {"validation": evaluate(y[valid], valid_predictions[name]), "test": evaluate(y[test], test_predictions[name])}
        for name in test_predictions
    }
    base_test = base.iloc[test].reset_index(drop=True)
    cohorts = cohort_metrics(base_test, y[test], test_predictions)
    timer.mark("evaluate")

    runtime = {
        "seconds": timer.total(),
        "window": window,
        "delay": args.delay,
        "stage_seconds": timer.stages,
        "green_runtime": green_runtime,
        "sd_runtime": sd_runtime,
        "model_fit_timing": model_timing,
        "permutation_seed": args.seed + 1009 * window,
        "history_labels_permuted": True,
        "target_labels_true": True,
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
    }
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    save_json(out / "runtime.json", runtime)

    pred = pd.DataFrame(
        {
            "TransactionID": data.iloc[test]["TransactionID"].to_numpy(),
            "TransactionDT": data.iloc[test]["TransactionDT"].to_numpy(),
            "isFraud": y[test],
            **{f"score_{name}": score for name, score in test_predictions.items()},
        }
    )
    pred.to_parquet(out / "predictions_test.parquet", index=False)

    rows = []
    for model, payload in metrics.items():
        rows.append(
            {
                "window": window,
                "model": model,
                "runtime_seconds": runtime["seconds"],
                **{metric: payload["test"].get(metric, np.nan) for metric in METRICS},
            }
        )
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({row["model"]: row for row in rows}, indent=2), flush=True)
    return {"summary": rows, "cohorts": cohorts, "runtime": runtime}


def aggregate_gains(summary: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    baseline = summary[summary["model"] == baseline_model].set_index("window")
    rows = []
    for model, group in summary.groupby("model"):
        if model == baseline_model:
            continue
        group = group.set_index("window").sort_index()
        common = group.index.intersection(baseline.index)
        out = {"model": model, "baseline": baseline_model, "windows": int(len(common))}
        for metric in METRICS:
            delta = group.loc[common, metric] - baseline.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(delta.mean())
            out[f"win_count_{metric}"] = int((delta > 0).sum())
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mean_gain_precision_at_0.01", "mean_gain_auc_pr"], ascending=False)


def compare_to_real(placebo_root: Path, real_delay_root: Path) -> pd.DataFrame:
    placebo = pd.read_csv(placebo_root / "window_summary.csv")
    real = pd.read_csv(real_delay_root / "per_delay_window_summary.csv")
    real = real[real["delay"] == 0].copy()
    rows = []
    for baseline in ("M3", "M3_H_raw"):
        placebo_gains = aggregate_gains(placebo, baseline)
        real_gains = aggregate_gains(real, baseline)
        merged = placebo_gains.merge(real_gains, on=["model", "baseline"], suffixes=("_placebo", "_real"))
        for _, row in merged.iterrows():
            out = {"model": row["model"], "baseline": baseline}
            for metric in ("auc_pr", "precision_at_0.01"):
                pg = float(row[f"mean_gain_{metric}_placebo"])
                rg = float(row[f"mean_gain_{metric}_real"])
                out[f"real_mean_gain_{metric}"] = rg
                out[f"placebo_mean_gain_{metric}"] = pg
                out[f"collapse_{metric}"] = rg - pg
                out[f"real_win_count_{metric}"] = int(row[f"win_count_{metric}_real"])
                out[f"placebo_win_count_{metric}"] = int(row[f"win_count_{metric}_placebo"])
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["baseline", "collapse_precision_at_0.01"], ascending=[True, False])


def write_aggregate_outputs(root: Path, real_delay_root: Path) -> None:
    placebo_root = root / RUN_SUBDIR
    summaries = []
    cohorts = []
    runtimes = {}
    for path in sorted(placebo_root.glob("window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(placebo_root.glob("window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            row["window"] = window
            cohorts.append(row)
    for path in sorted(placebo_root.glob("window_*/runtime.json")):
        runtimes[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))

    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    summary.to_csv(placebo_root / "window_summary.csv", index=False)
    cohort_frame.to_csv(placebo_root / "cohort_metrics.csv", index=False)
    save_json(placebo_root / "runtime.json", runtimes)
    aggregate_gains(summary, "M3").to_csv(placebo_root / "mean_gains_vs_m3.csv", index=False)
    aggregate_gains(summary, "M3_H_raw").to_csv(placebo_root / "mean_gains_vs_history.csv", index=False)
    comparison = compare_to_real(placebo_root, real_delay_root)
    comparison.to_csv(placebo_root / "permutation_placebo_summary.csv", index=False)

    report = [
        "# Run 3b: block-causal label-permutation placebo",
        "",
        "History labels used to compute H, S_D, and tail meta-features were permuted within each 100k window.",
        "True labels used for training/evaluation were not permuted.",
        "",
        "## Main placebo comparison",
        "",
        comparison.to_csv(index=False),
    ]
    (placebo_root / "permutation_placebo_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/00_moderate_100k_block_causal")
    parser.add_argument("--real-delay-root", default="outputs/ieee_green_final_review_gates_v2_block_causal/02_delay_sweep")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--precision-alpha", type=float, default=5.0)
    parser.add_argument("--ego-radius", type=int, default=2)
    parser.add_argument("--max-ego-nodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-warmup-fraction", type=float, default=0.5)
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--force-oof", action="store_true")
    parser.add_argument("--include-crossfit", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    (root / RUN_SUBDIR).mkdir(parents=True, exist_ok=True)
    if not args.aggregate_existing:
        summaries, cohorts, runtimes = [], [], {}
        for window in parse_ints(args.windows):
            result = run_window(args, window)
            summaries.extend(result["summary"])
            for row in result["cohorts"]:
                row["window"] = window
            cohorts.extend(result["cohorts"])
            runtimes[f"window_{window}"] = result["runtime"]
        # Window-level files are already written; aggregate from disk so resume and
        # fresh modes use identical code paths.
    write_aggregate_outputs(root, Path(args.real_delay_root))


if __name__ == "__main__":
    main()
