from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import evaluate, save_json
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached
from run_green_adaptive_theory import PrecisionWeightedGreenBuilder
from run_green_crossfit_reranker import crossfit_logistic_tail, oof_base_predictions
from run_green_focused_improvements import tuned_soft_mixture
from run_green_reranker_upgrade import logistic_tail_reranker
from run_green_risk_field import base_groups, cohort_metrics, green_columns


RUN_SUBDIR = "02_delay_sweep"
METRICS = [
    "auc_pr",
    "roc_auc",
    "precision_at_0.005",
    "precision_at_0.01",
    "precision_at_0.02",
    "precision_at_0.05",
]


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def delay_dir(root: Path, delay: int, window: int) -> Path:
    return root / RUN_SUBDIR / f"delay_{delay}" / f"window_{window}"


def ensure_delay_sd(
    args: argparse.Namespace,
    data: pd.DataFrame,
    delay: int,
    window: int,
    out: Path,
) -> tuple[Path, dict]:
    sd_path = out / "precision_weighted_green.parquet"
    if sd_path.exists() and not args.force_sd:
        return sd_path, {"cached": True, "path": str(sd_path)}

    reference_path = Path(args.adaptive_dir) / f"window_{window}" / "precision_weighted_green.parquet"
    if delay == 0 and reference_path.exists() and not args.force_sd:
        shutil.copy2(reference_path, sd_path)
        return sd_path, {"copied_from_reference": True, "path": str(sd_path)}

    builder = PrecisionWeightedGreenBuilder(
        delay_days=delay,
        history_alpha=args.alpha,
        precision_alpha=args.precision_alpha,
        radius=args.ego_radius,
        cap=args.max_ego_nodes,
    )
    runtime = builder.write_parquet(data, sd_path, batch_size=args.batch_size)
    runtime["cached"] = False
    runtime["path"] = str(sd_path)
    return sd_path, runtime


def load_delay_frame(args: argparse.Namespace, delay: int, window: int, data: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict]:
    baseline_window = Path(args.baseline_dir) / f"window_{window}"
    base = pd.read_parquet(baseline_window / "base_features.parquet").reset_index(drop=True)
    green_path = baseline_window / "green_features.parquet"
    schema = pq.ParquetFile(green_path).schema.names
    hcols = green_columns(schema, delay, args.alpha, "H")
    if not hcols:
        raise ValueError(f"No H columns found for delay={delay}, alpha={args.alpha}, window={window}")
    green = pd.read_parquet(green_path, columns=hcols).reset_index(drop=True)
    sd_path, sd_runtime = ensure_delay_sd(args, data, delay, window, out)
    sd_schema = pq.ParquetFile(sd_path).schema.names
    sd_sqrt_cols = [c for c in sd_schema if "SD_sqrt" in c]
    if not sd_sqrt_cols:
        raise ValueError(f"No SD_sqrt columns found in {sd_path}")
    sd = pd.read_parquet(sd_path, columns=sd_sqrt_cols).reset_index(drop=True)
    frame = pd.concat([base, green, sd], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return base, frame, hcols, sd_sqrt_cols, sd_runtime


def summary_columns(hcols: list[str], sd_cols: list[str]) -> list[str]:
    h_summary = [
        c
        for c in hcols
        if c.endswith("__H_left")
        or c.endswith("__H_right")
        or c.endswith("__H_left_exposure_log")
        or c.endswith("__H_right_exposure_log")
        or c.endswith("__H_left_missing")
        or c.endswith("__H_right_missing")
    ]
    sd_summary = [c for c in sd_cols if c.startswith("agg__") or "_all_" in c or c.endswith("_max") or c.endswith("_absdiff")]
    return list(dict.fromkeys(h_summary + sd_summary))


def run_delay_window(args: argparse.Namespace, delay: int, window: int) -> dict:
    timer = StageTimer()
    root = Path(args.out_dir)
    out = delay_dir(root, delay, window)
    out.mkdir(parents=True, exist_ok=True)

    data = load_ieee_cis_cached(
        args.data_dir,
        (window + 1) * args.window_size,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    )
    data = data.iloc[window * args.window_size : (window + 1) * args.window_size].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    timer.mark("load_data")

    base, frame, hcols, sd_cols, sd_runtime = load_delay_frame(args, delay, window, data, out)
    timer.mark("load_or_compute_delay_features")

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
    )
    valid_predictions["adaptive_two_stage"] = two_valid
    test_predictions["adaptive_two_stage"] = two_test
    selection["adaptive_two_stage"] = two_selection
    timer.mark("adaptive_two_stage")

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
    )
    valid_predictions["crossfit_logistic_tail"] = cross_valid
    test_predictions["crossfit_logistic_tail"] = cross_test
    selection["crossfit_logistic_tail"] = cross_selection
    selection["oof_base_predictions"] = oof_details
    timer.mark("crossfit")

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
        "delay": delay,
        "stage_seconds": timer.stages,
        "sd_runtime": sd_runtime,
        "model_fit_timing": model_timing,
        "uses_cached_base_green": True,
        "recomputed_delay_adaptive_precision": not bool(sd_runtime.get("cached") or sd_runtime.get("copied_from_reference")),
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
                "delay": delay,
                "window": window,
                "model": model,
                "runtime_seconds": runtime["seconds"],
                **{metric: payload["test"].get(metric, np.nan) for metric in METRICS},
            }
        )
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({row["model"]: row for row in rows}, indent=2))
    return {"summary": rows, "cohorts": cohorts, "runtime": runtime}


def aggregate_gains(summary: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    rows = []
    for delay, delay_rows in summary.groupby("delay"):
        baseline = delay_rows[delay_rows["model"] == baseline_model].set_index("window")
        for model, group in delay_rows.groupby("model"):
            if model == baseline_model:
                continue
            group = group.set_index("window").sort_index()
            common = group.index.intersection(baseline.index)
            out = {"delay": delay, "model": model, "baseline": baseline_model, "windows": int(len(common))}
            for metric in METRICS:
                delta = group.loc[common, metric] - baseline.loc[common, metric]
                out[f"mean_gain_{metric}"] = float(delta.mean())
                out[f"win_count_{metric}"] = int((delta > 0).sum())
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["delay", "mean_gain_precision_at_0.01"], ascending=[True, False])


def graph_marginal(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("graph_marginal", "M3_H_raw_S_D", "M3_H_raw"),
        ("soft_vs_history", "adaptive_soft", "M3_H_raw"),
        ("two_stage_vs_history", "adaptive_two_stage", "M3_H_raw"),
        ("crossfit_vs_history", "crossfit_logistic_tail", "M3_H_raw"),
    ]
    for delay, delay_rows in summary.groupby("delay"):
        for comparison, model, baseline_model in comparisons:
            model_rows = delay_rows[delay_rows["model"] == model].set_index("window")
            base_rows = delay_rows[delay_rows["model"] == baseline_model].set_index("window")
            common = model_rows.index.intersection(base_rows.index)
            out = {"delay": delay, "comparison": comparison, "model": model, "baseline": baseline_model, "windows": int(len(common))}
            for metric in METRICS:
                delta = model_rows.loc[common, metric] - base_rows.loc[common, metric]
                out[f"mean_gain_{metric}"] = float(delta.mean())
                out[f"win_count_{metric}"] = int((delta > 0).sum())
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["delay", "comparison"])


def aggregate_cohort_gains(cohorts: pd.DataFrame, baseline_model: str = "M3_H_raw") -> pd.DataFrame:
    rows = []
    critical = cohorts[cohorts["cohort"].isin(["known_endpoints", "C00_newedge"])].copy()
    for (delay, cohort), cohort_rows in critical.groupby(["delay", "cohort"]):
        baseline = cohort_rows[cohort_rows["model"] == baseline_model].set_index("window")
        for model, group in cohort_rows.groupby("model"):
            if model == baseline_model:
                continue
            group = group.set_index("window").sort_index()
            common = group.index.intersection(baseline.index)
            if len(common) == 0:
                continue
            out = {"delay": delay, "cohort": cohort, "model": model, "baseline": baseline_model, "windows": int(len(common))}
            for metric in ["auc_pr", "precision_at_0.01"]:
                delta = group.loc[common, metric] - baseline.loc[common, metric]
                out[f"mean_gain_{metric}"] = float(delta.mean())
                out[f"win_count_{metric}"] = int((delta > 0).sum())
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["delay", "cohort", "mean_gain_precision_at_0.01"], ascending=[True, True, False])


def write_aggregate_outputs(root: Path) -> None:
    sweep_root = root / RUN_SUBDIR
    summaries = []
    cohorts = []
    runtimes = {}
    for path in sorted(sweep_root.glob("delay_*/window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(sweep_root.glob("delay_*/window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        delay = int(path.parent.parent.name.split("_")[-1])
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            row["window"] = window
            row["delay"] = delay
            cohorts.append(row)
    for path in sorted(sweep_root.glob("delay_*/window_*/runtime.json")):
        runtimes[f"{path.parent.parent.name}/{path.parent.name}"] = json.loads(path.read_text(encoding="utf-8"))

    summary = pd.DataFrame(summaries).sort_values(["delay", "window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["delay", "window", "cohort", "model"])
    summary.to_csv(sweep_root / "per_delay_window_summary.csv", index=False)
    cohort_frame.to_csv(sweep_root / "cohort_metrics.csv", index=False)
    aggregate_gains(summary, "M3").to_csv(sweep_root / "per_delay_mean_gains_vs_m3.csv", index=False)
    aggregate_gains(summary, "M3_H_raw").to_csv(sweep_root / "per_delay_mean_gains_vs_history.csv", index=False)
    graph_marginal(summary).to_csv(sweep_root / "delay_graph_marginal_summary.csv", index=False)
    aggregate_cohort_gains(cohort_frame).to_csv(sweep_root / "critical_cohort_delay_gains.csv", index=False)
    save_json(sweep_root / "runtime.json", runtimes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--adaptive-dir", default="outputs/ieee_green_adaptive_theory_v1")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--precision-alpha", type=float, default=5.0)
    parser.add_argument("--ego-radius", type=int, default=2)
    parser.add_argument("--max-ego-nodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--force-sd", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--force-oof", action="store_true")
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-warmup-fraction", type=float, default=0.5)
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    (root / RUN_SUBDIR).mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        write_aggregate_outputs(root)
        return

    for delay in parse_ints(args.delays):
        for window in parse_ints(args.windows):
            run_delay_window(args, delay, window)
    write_aggregate_outputs(root)


if __name__ == "__main__":
    main()

