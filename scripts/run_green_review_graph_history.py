from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import evaluate, save_json
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached
from run_green_risk_field import base_groups, cohort_metrics, green_columns


METRICS = [
    "auc_pr",
    "roc_auc",
    "precision_at_0.005",
    "precision_at_0.01",
    "precision_at_0.02",
    "precision_at_0.05",
]

RUN_SUBDIR = "01_graph_vs_history"


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def safe_eval(y_true: np.ndarray, score: np.ndarray) -> dict[str, float] | None:
    if len(y_true) < 20 or np.unique(y_true).size < 2:
        return None
    return evaluate(y_true, score)


def load_reference_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(path)
    required = {
        "TransactionID",
        "isFraud",
        "score_adaptive_soft_current",
        "score_split_valid_logistic_tail",
        "score_crossfit_logistic_tail",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise FileNotFoundError(f"{path} is missing required reference columns: {missing}")
    return predictions


def exposure_counts(green: pd.DataFrame, hcols: list[str]) -> pd.DataFrame:
    left_cols = [c for c in hcols if c.endswith("__H_left_exposure_log")]
    right_cols = [c for c in hcols if c.endswith("__H_right_exposure_log")]
    if not left_cols or not right_cols:
        raise ValueError("No H exposure columns found; cannot build count strata")
    left = np.expm1(green[left_cols].astype(float)).clip(lower=0).sum(axis=1)
    right = np.expm1(green[right_cols].astype(float)).clip(lower=0).sum(axis=1)
    output = pd.DataFrame(
        {
            "released_count_left_sum": left.astype("float32"),
            "released_count_right_sum": right.astype("float32"),
        },
        index=green.index,
    )
    output["released_count_min"] = np.minimum(output["released_count_left_sum"], output["released_count_right_sum"])
    output["released_count_max"] = np.maximum(output["released_count_left_sum"], output["released_count_right_sum"])
    output["released_count_sum"] = output["released_count_left_sum"] + output["released_count_right_sum"]
    return output


def count_strata_masks(base_test: pd.DataFrame, exposure_test: pd.DataFrame) -> dict[str, np.ndarray]:
    min_count = exposure_test["released_count_min"].to_numpy(float)
    left = exposure_test["released_count_left_sum"].to_numpy(float)
    right = exposure_test["released_count_right_sum"].to_numpy(float)
    masks = {
        "cold_or_new": (left <= 0) | (right <= 0),
        "low_count": (min_count >= 1) & (min_count <= 5),
        "medium_count": (min_count >= 6) & (min_count <= 20),
        "high_count": min_count > 20,
        "known_endpoints": base_test["known_endpoints_all_edges"].to_numpy(bool),
        "C00_newedge": base_test["cohort_any_C00_newedge"].to_numpy(bool),
    }
    return masks


def summarize_strata(
    window: int,
    base_test: pd.DataFrame,
    exposure_test: pd.DataFrame,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for stratum, mask in count_strata_masks(base_test, exposure_test).items():
        y_mask = y_test[mask]
        if len(y_mask) < 20 or np.unique(y_mask).size < 2:
            continue
        for model, score in predictions.items():
            metrics = safe_eval(y_mask, score[mask])
            if metrics is None:
                continue
            rows.append(
                {
                    "window": window,
                    "stratum": stratum,
                    "model": model,
                    "rows": int(mask.sum()),
                    "frauds": int(y_mask.sum()),
                    "fraud_prevalence": float(y_mask.mean()),
                    "released_count_min_mean": float(exposure_test.loc[mask, "released_count_min"].mean()),
                    "released_count_sum_mean": float(exposure_test.loc[mask, "released_count_sum"].mean()),
                    **metrics,
                }
            )
    return rows


def summary_rows(window: int, metrics: dict[str, dict], runtime_seconds: float) -> list[dict]:
    rows = []
    for model, payload in metrics.items():
        test_metrics = payload["test"]
        rows.append(
            {
                "window": window,
                "model": model,
                "runtime_seconds": runtime_seconds,
                **{metric: test_metrics.get(metric, np.nan) for metric in METRICS},
            }
        )
    return rows


def run_window(args: argparse.Namespace, window: int) -> dict:
    timer = StageTimer()
    root = Path(args.out_dir)
    out = root / RUN_SUBDIR / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)

    baseline_window = Path(args.baseline_dir) / f"window_{window}"
    adaptive_window = Path(args.adaptive_dir) / f"window_{window}"
    crossfit_window = Path(args.crossfit_dir) / f"window_{window}"

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

    base = pd.read_parquet(baseline_window / "base_features.parquet").reset_index(drop=True)
    green_path = baseline_window / "green_features.parquet"
    schema = pq.ParquetFile(green_path).schema.names
    hcols = green_columns(schema, args.delay, args.alpha, "H")
    scols = green_columns(schema, args.delay, args.alpha, "S")
    green = pd.read_parquet(green_path, columns=hcols + scols).reset_index(drop=True)
    exposure = exposure_counts(green, hcols)
    timer.mark("read_base_green")

    sd_schema = pq.ParquetFile(adaptive_window / "precision_weighted_green.parquet").schema.names
    sd_cols_all = [c for c in sd_schema if "__SD_" in c or "__SD" in c]
    sd_sqrt_cols = [c for c in sd_cols_all if "SD_sqrt" in c]
    sd = pd.read_parquet(adaptive_window / "precision_weighted_green.parquet", columns=sd_sqrt_cols).reset_index(drop=True)
    timer.mark("read_adaptive_green")

    frame = pd.concat([base, green, sd], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    baseline_cols = base_groups(list(base.columns))["BP"]
    model_cols = {
        "M3": baseline_cols,
        "M3_H_raw": baseline_cols + hcols,
        "M3_S_plain": baseline_cols + scols,
        "M3_H_raw_S_plain": baseline_cols + hcols + scols,
        "M3_S_D": baseline_cols + sd_sqrt_cols,
        "M3_H_raw_S_D": baseline_cols + hcols + sd_sqrt_cols,
    }
    timer.mark("assemble_models")

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
    timer.mark("fit_ablation_models")

    reference = load_reference_predictions(crossfit_window / "predictions_test.parquet")
    expected_ids = data.iloc[test]["TransactionID"].to_numpy()
    if not np.array_equal(reference["TransactionID"].to_numpy(), expected_ids):
        raise ValueError(f"Reference predictions for window {window} do not align with the chronological test slice")
    test_predictions["adaptive_soft"] = reference["score_adaptive_soft_current"].to_numpy(float)
    test_predictions["adaptive_two_stage"] = reference["score_split_valid_logistic_tail"].to_numpy(float)
    test_predictions["crossfit_logistic_tail"] = reference["score_crossfit_logistic_tail"].to_numpy(float)
    reference_selection = {
        "adaptive_soft": {"source": str(crossfit_window / "predictions_test.parquet"), "test_used_for_selection": False},
        "adaptive_two_stage": {"source": str(crossfit_window / "predictions_test.parquet"), "test_used_for_selection": False},
        "crossfit_logistic_tail": {"source": str(crossfit_window / "predictions_test.parquet"), "test_used_for_selection": False},
    }
    selection.update(reference_selection)
    timer.mark("load_reference_scores")

    metrics = {}
    for model, score in test_predictions.items():
        if model in valid_predictions:
            validation = evaluate(y[valid], valid_predictions[model])
        else:
            validation = {"reference_score_only": True}
        metrics[model] = {"validation": validation, "test": evaluate(y[test], score)}
    base_test = base.iloc[test].reset_index(drop=True)
    exposure_test = exposure.iloc[test].reset_index(drop=True)
    y_test = y[test]
    cohorts = cohort_metrics(base_test, y_test, test_predictions)
    strata = summarize_strata(window, base_test, exposure_test, y_test, test_predictions)
    timer.mark("evaluate")

    runtime = {
        "seconds": timer.total(),
        "window": window,
        "stage_seconds": timer.stages,
        "model_fit_timing": model_timing,
        "uses_cached_dense_exact_green": True,
        "uses_cached_adaptive_precision_green": True,
        "reference_adaptive_scores_loaded": True,
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
    }
    rows = summary_rows(window, metrics, runtime["seconds"])
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "count_strata_metrics.csv", index=False)
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    save_json(out / "runtime.json", runtime)
    pd.DataFrame({"TransactionID": expected_ids, **{f"score_{k}": v for k, v in test_predictions.items()}}).to_parquet(
        out / "predictions_test.parquet", index=False
    )

    print(json.dumps({row["model"]: row for row in rows}, indent=2))
    return {"summary": rows, "cohorts": cohorts, "strata": strata, "runtime": runtime}


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


def aggregate_strata(strata: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    rows = []
    for (stratum, model), group in strata.groupby(["stratum", "model"]):
        if model == baseline_model:
            continue
        base = strata[(strata["stratum"] == stratum) & (strata["model"] == baseline_model)].set_index("window")
        group = group.set_index("window").sort_index()
        common = group.index.intersection(base.index)
        if len(common) == 0:
            continue
        out = {
            "stratum": stratum,
            "model": model,
            "baseline": baseline_model,
            "windows": int(len(common)),
            "mean_rows": float(group.loc[common, "rows"].mean()),
            "mean_fraud_prevalence": float(group.loc[common, "fraud_prevalence"].mean()),
        }
        for metric in ["auc_pr", "precision_at_0.01"]:
            delta = group.loc[common, metric] - base.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(delta.mean())
            out[f"win_count_{metric}"] = int((delta > 0).sum())
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["stratum", "mean_gain_precision_at_0.01"], ascending=[True, False])


def graph_marginal_summary(summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("plain_graph_marginal", "M3_H_raw_S_plain", "M3_H_raw"),
        ("adaptive_graph_marginal", "M3_H_raw_S_D", "M3_H_raw"),
        ("adaptive_sd_without_raw_vs_m3", "M3_S_D", "M3"),
        ("plain_s_without_raw_vs_m3", "M3_S_plain", "M3"),
        ("accepted_two_stage_vs_history", "adaptive_two_stage", "M3_H_raw"),
        ("crossfit_vs_history", "crossfit_logistic_tail", "M3_H_raw"),
    ]
    rows = []
    for comparison, model, baseline_model in comparisons:
        model_rows = summary[summary["model"] == model].set_index("window")
        base_rows = summary[summary["model"] == baseline_model].set_index("window")
        common = model_rows.index.intersection(base_rows.index)
        out = {"comparison": comparison, "model": model, "baseline": baseline_model, "windows": int(len(common))}
        for metric in METRICS:
            delta = model_rows.loc[common, metric] - base_rows.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(delta.mean())
            out[f"win_count_{metric}"] = int((delta > 0).sum())
        out["success_graph_value"] = bool(
            comparison == "adaptive_graph_marginal"
            and (
                (out["mean_gain_auc_pr"] > 0 and out["win_count_auc_pr"] >= 4)
                or (out["mean_gain_precision_at_0.01"] > 0 and out["win_count_precision_at_0.01"] >= 4)
            )
        )
        rows.append(out)
    return pd.DataFrame(rows)


def critical_cohort_graph_margins(cohorts: pd.DataFrame) -> pd.DataFrame:
    critical = cohorts[cohorts["cohort"].isin(["known_endpoints", "C00_newedge"])].copy()
    rows = []
    for cohort in sorted(critical["cohort"].unique()):
        for model, baseline_model in [
            ("M3_H_raw_S_D", "M3_H_raw"),
            ("adaptive_two_stage", "M3_H_raw"),
            ("crossfit_logistic_tail", "M3_H_raw"),
        ]:
            model_rows = critical[(critical["cohort"] == cohort) & (critical["model"] == model)].set_index("window")
            base_rows = critical[(critical["cohort"] == cohort) & (critical["model"] == baseline_model)].set_index("window")
            common = model_rows.index.intersection(base_rows.index)
            if len(common) == 0:
                continue
            out = {"cohort": cohort, "model": model, "baseline": baseline_model, "windows": int(len(common))}
            for metric in ["auc_pr", "precision_at_0.01"]:
                delta = model_rows.loc[common, metric] - base_rows.loc[common, metric]
                out[f"mean_gain_{metric}"] = float(delta.mean())
                out[f"win_count_{metric}"] = int((delta > 0).sum())
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["cohort", "mean_gain_precision_at_0.01"], ascending=[True, False])


def write_aggregate_outputs(root: Path) -> None:
    graph_root = root / RUN_SUBDIR
    summaries = []
    cohorts = []
    strata_rows = []
    runtimes = {}
    for path in sorted(graph_root.glob("window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(graph_root.glob("window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            row["window"] = window
            cohorts.append(row)
    for path in sorted(graph_root.glob("window_*/count_strata_metrics.csv")):
        if path.stat().st_size:
            strata_rows.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(graph_root.glob("window_*/runtime.json")):
        runtimes[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))

    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    strata = pd.DataFrame(strata_rows).sort_values(["window", "stratum", "model"])
    summary.to_csv(graph_root / "window_summary.csv", index=False)
    cohort_frame.to_csv(graph_root / "cohort_metrics.csv", index=False)
    strata.to_csv(graph_root / "count_strata_metrics.csv", index=False)
    save_json(graph_root / "runtime.json", runtimes)

    aggregate_gains(summary, "M3").to_csv(graph_root / "mean_gains_vs_m3.csv", index=False)
    aggregate_gains(summary, "M3_H_raw").to_csv(graph_root / "mean_gains_vs_history.csv", index=False)
    graph_marginal_summary(summary).to_csv(graph_root / "graph_marginal_summary.csv", index=False)
    aggregate_strata(strata, "M3_H_raw").to_csv(graph_root / "count_strata_mean_gains.csv", index=False)
    critical_cohort_graph_margins(cohort_frame).to_csv(graph_root / "critical_cohort_graph_margins.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--adaptive-dir", default="outputs/ieee_green_adaptive_theory_v1")
    parser.add_argument("--crossfit-dir", default="outputs/ieee_green_crossfit_reranker_v1")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    (root / RUN_SUBDIR).mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        write_aggregate_outputs(root)
        return

    for window in parse_ints(args.windows):
        run_window(args, window)
    write_aggregate_outputs(root)


if __name__ == "__main__":
    main()

