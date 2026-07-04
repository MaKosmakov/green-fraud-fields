from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, early_stopping
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis_cached
from green_fraud_fields.modeling import evaluate, save_json
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached
from run_green_adaptive_theory import add_adaptive_shrinkage, cohort_gate
from run_green_focused_improvements import selection_key, tuned_soft_mixture
from run_green_risk_field import base_groups, cohort_metrics, green_columns
from run_green_risk_tail import rerank


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def safe_logit(score: np.ndarray) -> np.ndarray:
    return logit(np.clip(score, 1e-6, 1 - 1e-6))


def meta_matrices(
    frame: pd.DataFrame,
    valid: slice,
    test: slice,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    raw_valid = frame.iloc[valid][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_test = frame.iloc[test][summary_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    meta_valid = np.column_stack([
        *[safe_logit(valid_predictions[name]) for name in model_names],
        imputer.fit_transform(raw_valid),
    ])
    meta_test = np.column_stack([
        *[safe_logit(test_predictions[name]) for name in model_names],
        imputer.transform(raw_test),
    ])
    return meta_valid, meta_test


def logistic_tail_reranker(
    frame: pd.DataFrame,
    y: np.ndarray,
    valid: slice,
    test: slice,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
    fractions: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_valid = y[valid]
    split = len(y_valid) // 2
    meta_valid, meta_test = meta_matrices(frame, valid, test, valid_predictions, test_predictions, model_names, summary_cols)
    best = None
    trials = []
    for fraction in fractions:
        first_base = valid_predictions["M3"][:split]
        train_mask = first_base >= np.quantile(first_base, 1 - fraction)
        if np.unique(y_valid[:split][train_mask]).size < 2:
            continue
        model = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0)
        model.fit(meta_valid[:split][train_mask], y_valid[:split][train_mask])
        second_base = valid_predictions["M3"][split:]
        second_mask = second_base >= np.quantile(second_base, 1 - fraction)
        probability = model.predict_proba(meta_valid[split:])[:, 1]
        score = rerank(second_base, probability, second_mask)
        metrics = evaluate(y_valid[split:], score)
        key = (metrics["precision_at_0.01"], metrics["precision_at_0.005"], metrics["auc_pr"])
        trials.append({"fraction": fraction, "selection_key": list(key), "validation_half2": metrics})
        if best is None or key > best[0]:
            best = (key, fraction)
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in validation tail",
            "test_used_for_selection": False,
        }
    fraction = best[1]
    train_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    final = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0)
    final.fit(meta_valid[train_mask], y_valid[train_mask])
    valid_probability = final.predict_proba(meta_valid)[:, 1]
    test_probability = final.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    test_mask = test_predictions["M3"] >= np.quantile(test_predictions["M3"], 1 - fraction)
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "validation split-half P@1%, then P@0.5%, then AUC-PR",
        "candidate_fraction": fraction,
        "selection_key": list(best[0]),
        "trials": trials,
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


def lightgbm_tail_reranker(
    frame: pd.DataFrame,
    y: np.ndarray,
    valid: slice,
    test: slice,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
    fractions: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_valid = y[valid]
    split = len(y_valid) // 2
    meta_valid, meta_test = meta_matrices(frame, valid, test, valid_predictions, test_predictions, model_names, summary_cols)
    params_grid = [
        {"num_leaves": 7, "min_child_samples": 20, "reg_lambda": 5.0},
        {"num_leaves": 15, "min_child_samples": 30, "reg_lambda": 3.0},
        {"num_leaves": 31, "min_child_samples": 50, "reg_lambda": 8.0},
    ]
    best = None
    trials = []
    for fraction in fractions:
        first_base = valid_predictions["M3"][:split]
        train_mask = first_base >= np.quantile(first_base, 1 - fraction)
        if np.unique(y_valid[:split][train_mask]).size < 2:
            continue
        positives = max(float(y_valid[:split][train_mask].sum()), 1.0)
        scale = (float(train_mask.sum()) - positives) / positives
        for index, params in enumerate(params_grid):
            model = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale,
                random_state=seed + index,
                n_jobs=-1,
                verbosity=-1,
                **params,
            )
            model.fit(
                meta_valid[:split][train_mask],
                y_valid[:split][train_mask],
                eval_set=[(meta_valid[split:], y_valid[split:])],
                eval_metric="average_precision",
                callbacks=[early_stopping(30, verbose=False)],
            )
            second_base = valid_predictions["M3"][split:]
            second_mask = second_base >= np.quantile(second_base, 1 - fraction)
            probability = model.predict_proba(meta_valid[split:])[:, 1]
            score = rerank(second_base, probability, second_mask)
            metrics = evaluate(y_valid[split:], score)
            key = (metrics["precision_at_0.01"], metrics["precision_at_0.005"], metrics["auc_pr"])
            trial = {
                "fraction": fraction,
                "params": params,
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
                "selection_key": list(key),
                "validation_half2": metrics,
            }
            trials.append(trial)
            if best is None or key > best[0]:
                best = (key, fraction, params, int(model.best_iteration_ or model.n_estimators))
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in validation tail",
            "test_used_for_selection": False,
        }
    _, fraction, params, best_iteration = best
    train_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    positives = max(float(y_valid[train_mask].sum()), 1.0)
    scale = (float(train_mask.sum()) - positives) / positives
    final = LGBMClassifier(
        n_estimators=max(best_iteration, 10),
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    final.fit(meta_valid[train_mask], y_valid[train_mask])
    valid_probability = final.predict_proba(meta_valid)[:, 1]
    test_probability = final.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    test_mask = test_predictions["M3"] >= np.quantile(test_predictions["M3"], 1 - fraction)
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "LightGBM tail reranker selected on validation split-half P@1%, then P@0.5%, then AUC-PR",
        "candidate_fraction": fraction,
        "params": params,
        "best_iteration": best_iteration,
        "selection_key": list(best[0]),
        "trials": trials,
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


def save_predictions(out: Path, data: pd.DataFrame, y_test: np.ndarray, test: slice, predictions: dict[str, np.ndarray]) -> None:
    pred = pd.DataFrame({
        "TransactionID": data.iloc[test]["TransactionID"].to_numpy(),
        "TransactionDT": data.iloc[test]["TransactionDT"].to_numpy(),
        "isFraud": y_test,
        **{f"score_{name}": score for name, score in predictions.items()},
    })
    pred.to_parquet(out / "predictions_test.parquet", index=False)


def run_window(args, window: int) -> dict:
    timer = StageTimer()
    out = Path(args.out_dir) / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    baseline_window = Path(args.baseline_dir) / f"window_{window}"
    reference_window = Path(args.reference_dir) / f"window_{window}"
    data = load_ieee_cis_cached(args.data_dir, (window + 1) * args.window_size, cache_dir=args.data_cache_dir)
    data = data.iloc[window * args.window_size:(window + 1) * args.window_size].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    timer.mark("load_data")

    base = pd.read_parquet(baseline_window / "base_features.parquet").reset_index(drop=True)
    green_path = baseline_window / "green_features.parquet"
    schema = pq.ParquetFile(green_path).schema.names
    delays = parse_ints(args.delays)
    hcols = [c for delay in delays for c in green_columns(schema, delay, args.alpha, "H")]
    scols = [c for delay in delays for c in green_columns(schema, delay, args.alpha, "S")]
    green = pd.read_parquet(green_path, columns=hcols + scols)
    green, sg_cols = add_adaptive_shrinkage(green, delays, args.alpha)
    sd = pd.read_parquet(reference_window / "precision_weighted_green.parquet")
    sd_cols = [c for c in sd.columns if "__SD_" in c or "__SD" in c]
    sd = sd[sd_cols]
    frame = pd.concat([base, green, sd], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    timer.mark("read_features")

    baseline_cols = base_groups(list(base.columns))["BP"]
    count_cols = [c for c in sd_cols if "SD_count" in c]
    sqrt_cols = [c for c in sd_cols if "SD_sqrt" in c]
    model_cols = {
        "M3": baseline_cols,
        "H_S": baseline_cols + hcols + scols,
        "adaptive_shrinkage": baseline_cols + hcols + sg_cols,
        "precision_green_count": baseline_cols + hcols + count_cols,
        "precision_green_sqrt": baseline_cols + hcols + sqrt_cols,
    }
    valid_predictions, test_predictions, selection, model_timing = fit_named_models_cached(
        frame,
        y,
        train,
        valid,
        test,
        model_cols,
        args.seed,
        cache_dir=out / "model_prediction_cache",
        force_models=args.force_models,
    )
    timer.mark("fit_or_load_base_models")

    mix_valid, mix_test, mix_selection = tuned_soft_mixture(
        y[valid],
        valid_predictions,
        test_predictions,
        ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt"],
    )
    valid_predictions["adaptive_soft_current"] = mix_valid
    test_predictions["adaptive_soft_current"] = mix_test
    selection["adaptive_soft_current"] = mix_selection
    timer.mark("soft_mixture")

    summary_cols = [c for c in sg_cols + sd_cols if c.startswith("agg__") or "all_" in c]
    components = ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt", "adaptive_soft_current"]
    fractions = tuple(float(x) for x in args.tail_fractions.split(",") if x.strip())

    log_valid, log_test, log_selection = logistic_tail_reranker(
        frame, y, valid, test, valid_predictions, test_predictions, components, summary_cols, fractions
    )
    valid_predictions["logistic_tail_regions"] = log_valid
    test_predictions["logistic_tail_regions"] = log_test
    selection["logistic_tail_regions"] = log_selection
    timer.mark("logistic_tail")

    lgb_valid, lgb_test, lgb_selection = lightgbm_tail_reranker(
        frame, y, valid, test, valid_predictions, test_predictions, components, summary_cols, fractions, args.seed
    )
    valid_predictions["lgbm_tail_regions"] = lgb_valid
    test_predictions["lgbm_tail_regions"] = lgb_test
    selection["lgbm_tail_regions"] = lgb_selection
    timer.mark("lgbm_tail")

    blend_components = ["M3", "adaptive_soft_current", "logistic_tail_regions", "lgbm_tail_regions"]
    blend_valid, blend_test, blend_selection = tuned_soft_mixture(y[valid], valid_predictions, test_predictions, blend_components)
    valid_predictions["tail_soft_blend"] = blend_valid
    test_predictions["tail_soft_blend"] = blend_test
    selection["tail_soft_blend"] = blend_selection

    gate_valid, gate_test, gate_selection = cohort_gate(
        base,
        y[valid],
        valid_predictions,
        test_predictions,
        valid,
        test,
        ["adaptive_soft_current", "logistic_tail_regions", "lgbm_tail_regions", "tail_soft_blend"],
    )
    valid_predictions["cohort_gated_tail"] = gate_valid
    test_predictions["cohort_gated_tail"] = gate_test
    selection["cohort_gated_tail"] = gate_selection
    timer.mark("blend_and_gate")

    metrics = {
        name: {"validation": evaluate(y[valid], valid_predictions[name]), "test": evaluate(y[test], test_predictions[name])}
        for name in valid_predictions
    }
    cohorts = cohort_metrics(base.iloc[test].reset_index(drop=True), y[test], test_predictions)
    save_predictions(out, data, y[test], test, test_predictions)
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    timer.mark("evaluate_and_write")

    runtime = {
        "seconds": timer.total(),
        "window": window,
        "stage_seconds": timer.stages,
        "model_fit_timing": model_timing,
        "uses_cached_dense_exact_green": True,
        "uses_reference_adaptive_fields": True,
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
    }
    save_json(out / "runtime.json", runtime)
    rows = []
    for name, values in metrics.items():
        test_metrics = values["test"]
        rows.append({
            "window": window,
            "model": name,
            "auc_pr": test_metrics["auc_pr"],
            "precision_at_0.005": test_metrics["precision_at_0.005"],
            "precision_at_0.01": test_metrics["precision_at_0.01"],
            "precision_at_0.02": test_metrics["precision_at_0.02"],
            "precision_at_0.05": test_metrics["precision_at_0.05"],
            "runtime_seconds": runtime["seconds"],
        })
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({row["model"]: row for row in rows}, indent=2))
    return {"summary": rows, "cohorts": cohorts, "runtime": runtime}


def write_aggregate_outputs(root: Path, reference_dir: Path) -> None:
    summaries, cohorts, runtimes = [], [], {}
    for path in sorted(root.glob("window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(root.glob("window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        rows = json.loads(path.read_text())["rows"]
        for row in rows:
            row["window"] = window
        cohorts.extend(rows)
    for path in sorted(root.glob("window_*/runtime.json")):
        runtimes[path.parent.name] = json.loads(path.read_text())
    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    summary.to_csv(root / "window_summary.csv", index=False)
    cohort_frame.to_csv(root / "cohort_metrics.csv", index=False)
    save_json(root / "runtime.json", runtimes)

    reference = pd.read_csv(reference_dir / "window_summary.csv")
    reference = reference[reference["model"] == "adaptive_two_stage"].set_index("window")
    baseline = summary[summary["model"] == "M3"].set_index("window")
    rows = []
    for model, group in summary.groupby("model"):
        if model == "M3":
            continue
        group = group.set_index("window").sort_index()
        common = group.index.intersection(baseline.index)
        ref_common = group.index.intersection(reference.index)
        out = {"model": model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            gain = group.loc[common, metric] - baseline.loc[common, metric]
            ref_delta = group.loc[ref_common, metric] - reference.loc[ref_common, metric]
            out[f"mean_gain_{metric}"] = float(gain.mean())
            out[f"win_count_{metric}"] = int((gain > 0).sum())
            out[f"mean_delta_ref_{metric}"] = float(ref_delta.mean())
            out[f"win_count_ref_{metric}"] = int((ref_delta > 0).sum())
        out["accepted_vs_ref"] = bool(
            (out["mean_delta_ref_precision_at_0.01"] > 0 or out["mean_delta_ref_auc_pr"] > 0)
            and out["win_count_ref_precision_at_0.01"] >= 4
        )
        rows.append(out)
    pd.DataFrame(rows).sort_values("mean_delta_ref_precision_at_0.01", ascending=False).to_csv(
        root / "mean_gains_vs_reference.csv", index=False
    )

    ref_cohorts = pd.read_csv(reference_dir / "cohort_metrics.csv")
    ref_cohorts = ref_cohorts[ref_cohorts["model"] == "adaptive_two_stage"]
    critical = cohort_frame[cohort_frame["cohort"].isin(["known_endpoints", "C00_newedge"])]
    critical_rows = []
    for (cohort, model), group in critical.groupby(["cohort", "model"]):
        if model == "M3":
            continue
        base_group = critical[(critical["cohort"] == cohort) & (critical["model"] == "M3")].set_index("window")
        ref_group = ref_cohorts[ref_cohorts["cohort"] == cohort].set_index("window")
        group = group.set_index("window").sort_index()
        common = group.index.intersection(base_group.index)
        ref_common = group.index.intersection(ref_group.index)
        out = {"cohort": cohort, "model": model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            gain = group.loc[common, metric] - base_group.loc[common, metric]
            ref_delta = group.loc[ref_common, metric] - ref_group.loc[ref_common, metric]
            out[f"mean_gain_{metric}"] = float(gain.mean())
            out[f"win_count_{metric}"] = int((gain > 0).sum())
            out[f"mean_delta_ref_{metric}"] = float(ref_delta.mean())
            out[f"win_count_ref_{metric}"] = int((ref_delta > 0).sum())
        critical_rows.append(out)
    pd.DataFrame(critical_rows).sort_values(
        ["cohort", "mean_delta_ref_precision_at_0.01"], ascending=[True, False]
    ).to_csv(root / "critical_cohort_vs_reference.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--reference-dir", default="outputs/ieee_green_adaptive_theory_v1")
    parser.add_argument("--out-dir", default="outputs/ieee_green_reranker_upgrade_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--tail-fractions", default="0.025,0.05,0.10,0.20,0.30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        write_aggregate_outputs(root, Path(args.reference_dir))
        return
    for window in parse_ints(args.windows):
        run_window(args, window)
    write_aggregate_outputs(root, Path(args.reference_dir))


if __name__ == "__main__":
    main()

