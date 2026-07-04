from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis
from green_fraud_fields.modeling import evaluate, save_json
from run_green_risk_field import base_groups, cohort_metrics, green_columns
from run_green_risk_tail import rerank
from run_ieee_tail_specialized import fit_tail_selected


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def selection_key(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    metrics = evaluate(y, score)
    return (
        metrics["precision_at_0.01"],
        metrics["precision_at_0.005"],
        metrics["auc_pr"],
    )


def fit_named_models(
    frame: pd.DataFrame,
    y: np.ndarray,
    train: slice,
    valid: slice,
    test: slice,
    model_columns: dict[str, list[str]],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    valid_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    selection: dict = {}
    for name, columns in model_columns.items():
        valid_score, test_score, details, _ = fit_tail_selected(
            frame.iloc[train][columns], y[train],
            frame.iloc[valid][columns], y[valid],
            frame.iloc[test][columns], seed,
        )
        valid_predictions[name] = valid_score
        test_predictions[name] = test_score
        selection[name] = details
    return valid_predictions, test_predictions, selection


def tuned_soft_mixture(
    y_valid: np.ndarray,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    components: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    logits_valid = {
        name: logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6))
        for name in components
    }
    logits_test = {
        name: logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6))
        for name in components
    }
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    best = None
    for candidate in product(weights, repeat=len(components)):
        total = sum(candidate)
        if total == 0:
            continue
        valid_score = expit(
            sum(w * logits_valid[name] for w, name in zip(candidate, components))
            / total
        )
        key = selection_key(y_valid, valid_score)
        if best is None or key > best[0]:
            best = (key, candidate, valid_score)
    assert best is not None
    test_score = expit(
        sum(w * logits_test[name] for w, name in zip(best[1], components))
        / sum(best[1])
    )
    return best[2], test_score, {
        "selection": "validation lexicographic P@1%, P@0.5%, AUC-PR",
        "components": components,
        "weights": dict(zip(components, best[1])),
        "validation": evaluate(y_valid, best[2]),
        "test_used_for_selection": False,
    }


def add_delay_contrasts(
    green: pd.DataFrame,
    delays: tuple[int, ...],
    alpha: int,
) -> tuple[pd.DataFrame, list[str]]:
    contrasts: dict[str, pd.Series] = {}
    pairs = ((0, 7), (1, 14))
    for block in ("H", "S"):
        for left, right in pairs:
            if left not in delays or right not in delays:
                continue
            left_marker = f"d{left}_a{alpha}__{block}_"
            right_marker = f"d{right}_a{alpha}__{block}_"
            for left_col in [c for c in green.columns if left_marker in c]:
                right_col = left_col.replace(left_marker, right_marker)
                if right_col not in green.columns:
                    continue
                name = (
                    "contrast__"
                    + left_col.replace(left_marker, f"d{left}_minus_d{right}_a{alpha}__{block}_")
                )
                contrasts[name] = green[left_col] - green[right_col]
    if not contrasts:
        return green, []
    contrast_frame = pd.DataFrame(contrasts, index=green.index).astype("float32")
    return pd.concat([green, contrast_frame], axis=1), list(contrast_frame.columns)


def tuned_two_stage(
    frame: pd.DataFrame,
    y: np.ndarray,
    valid: slice,
    test: slice,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_valid = y[valid]
    split = len(y_valid) // 2
    logits_valid = [
        logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6))
        for name in model_names
    ]
    logits_test = [
        logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6))
        for name in model_names
    ]
    raw_valid = frame.iloc[valid][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_test = frame.iloc[test][summary_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    meta_valid = np.column_stack([*logits_valid, imputer.fit_transform(raw_valid)])
    meta_test = np.column_stack([*logits_test, imputer.transform(raw_test)])
    best = None
    for fraction in (0.05, 0.10, 0.20, 0.30):
        first_base = valid_predictions["M3"][:split]
        train_mask = first_base >= np.quantile(first_base, 1 - fraction)
        if np.unique(y_valid[:split][train_mask]).size < 2:
            continue
        model = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=2000, random_state=0
        )
        model.fit(meta_valid[:split][train_mask], y_valid[:split][train_mask])
        second_base = valid_predictions["M3"][split:]
        second_mask = second_base >= np.quantile(second_base, 1 - fraction)
        second_probability = model.predict_proba(meta_valid[split:])[:, 1]
        second_score = rerank(second_base, second_probability, second_mask)
        key = (
            evaluate(y_valid[split:], second_score)["precision_at_0.01"],
            evaluate(y_valid[split:], second_score)["precision_at_0.005"],
            evaluate(y_valid[split:], second_score)["auc_pr"],
        )
        if best is None or key > best[0]:
            best = (key, fraction)
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in validation tail candidates",
            "test_used_for_selection": False,
        }
    fraction = best[1]
    train_mask = valid_predictions["M3"] >= np.quantile(
        valid_predictions["M3"], 1 - fraction
    )
    final = LogisticRegression(
        class_weight="balanced", C=0.5, max_iter=2000, random_state=0
    )
    final.fit(meta_valid[train_mask], y_valid[train_mask])
    valid_probability = final.predict_proba(meta_valid)[:, 1]
    test_probability = final.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= np.quantile(
        valid_predictions["M3"], 1 - fraction
    )
    test_mask = test_predictions["M3"] >= np.quantile(
        test_predictions["M3"], 1 - fraction
    )
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "validation split-half P@1%, then P@0.5%, then AUC-PR",
        "candidate_fraction": fraction,
        "selection_key": list(best[0]),
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


def run_window(args, window: int) -> dict:
    started = time.time()
    root = Path(args.baseline_dir)
    out = Path(args.out_dir) / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    baseline_window = root / f"window_{window}"
    if not baseline_window.exists():
        raise FileNotFoundError(baseline_window)
    data = load_ieee_cis(args.data_dir, (window + 1) * args.window_size)
    data = data.iloc[window * args.window_size:(window + 1) * args.window_size].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    base = pd.read_parquet(baseline_window / "base_features.parquet").reset_index(drop=True)
    parquet = pq.ParquetFile(baseline_window / "green_features.parquet")
    schema = parquet.schema.names
    delays = parse_ints(args.delays)
    alpha = args.alpha
    h_by_delay = {delay: green_columns(schema, delay, alpha, "H") for delay in delays}
    s_by_delay = {delay: green_columns(schema, delay, alpha, "S") for delay in delays}
    tr_by_delay = {delay: green_columns(schema, delay, alpha, "TR") for delay in delays}
    hcols = [c for delay in delays for c in h_by_delay[delay]]
    scols = [c for delay in delays for c in s_by_delay[delay]]
    trcols = [c for delay in delays for c in tr_by_delay[delay]]
    green = pd.read_parquet(
        baseline_window / "green_features.parquet",
        columns=hcols + scols + trcols,
    )
    green, contrast_cols = add_delay_contrasts(green, delays, alpha)
    frame = pd.concat([base, green], axis=1)
    baseline_cols = base_groups(list(base.columns))["BP"]
    single_delay_columns = {
        f"single_H_S_d{delay}": baseline_cols + h_by_delay[delay] + s_by_delay[delay]
        for delay in delays
    }
    core_columns = {
        "M3": baseline_cols,
        "H": baseline_cols + hcols,
        "H_S": baseline_cols + hcols + scols,
        "H_S_TR": baseline_cols + hcols + scols + trcols,
        "multi_delay_H_S_contrast": baseline_cols + hcols + scols + contrast_cols,
        **single_delay_columns,
    }
    valid_predictions, test_predictions, selection = fit_named_models(
        frame, y, train, valid, test, core_columns, args.seed
    )
    best_single = max(
        single_delay_columns,
        key=lambda name: selection_key(y[valid], valid_predictions[name]),
    )
    valid_predictions["single_best_delay_H_S"] = valid_predictions[best_single]
    test_predictions["single_best_delay_H_S"] = test_predictions[best_single]
    selection["single_best_delay_H_S"] = {
        "chosen_model": best_single,
        "validation": evaluate(y[valid], valid_predictions[best_single]),
        "test_used_for_selection": False,
    }
    mix_valid, mix_test, mix_selection = tuned_soft_mixture(
        y[valid],
        valid_predictions,
        test_predictions,
        ["M3", "H", "H_S", "H_S_TR"],
    )
    valid_predictions["soft_mixture_plus_TR"] = mix_valid
    test_predictions["soft_mixture_plus_TR"] = mix_test
    selection["soft_mixture_plus_TR"] = mix_selection
    summary_cols = [
        c for c in scols + trcols + contrast_cols
        if c.startswith("agg__") or c.startswith("contrast__agg__")
        or c.endswith(("__S_max", "__S_absdiff", "__TR_max_abs_update"))
    ]
    two_valid, two_test, two_selection = tuned_two_stage(
        frame,
        y,
        valid,
        test,
        valid_predictions,
        test_predictions,
        ["M3", "H", "H_S", "H_S_TR", "soft_mixture_plus_TR", "multi_delay_H_S_contrast"],
        summary_cols,
    )
    valid_predictions["two_stage_tail_p1"] = two_valid
    test_predictions["two_stage_tail_p1"] = two_test
    selection["two_stage_tail_p1"] = two_selection
    report_models = [
        "M3",
        "H",
        "H_S",
        "H_S_TR",
        "soft_mixture_plus_TR",
        "single_best_delay_H_S",
        "multi_delay_H_S_contrast",
        "two_stage_tail_p1",
    ]
    metrics = {
        name: {
            "validation": evaluate(y[valid], valid_predictions[name]),
            "test": evaluate(y[test], test_predictions[name]),
        }
        for name in report_models
    }
    test_base = base.iloc[test].reset_index(drop=True)
    cohorts = cohort_metrics(
        test_base,
        y[test],
        {name: test_predictions[name] for name in report_models},
    )
    predictions = pd.DataFrame({
        "TransactionID": data.iloc[test]["TransactionID"].to_numpy(),
        "TransactionDT": data.iloc[test]["TransactionDT"].to_numpy(),
        "isFraud": y[test],
        "window": window,
        **{f"score_{name}": test_predictions[name] for name in report_models},
    })
    predictions.to_parquet(out / "predictions_test.parquet", index=False)
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    baseline_runtime = json.loads((baseline_window / "runtime.json").read_text())
    runtime = {
        "seconds": time.time() - started,
        "window": window,
        "uses_cached_dense_exact_features": True,
        "baseline_feature_runtime": baseline_runtime.get("feature_runtime", {}),
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
    }
    save_json(out / "runtime.json", runtime)
    summary_rows = []
    for name in report_models:
        test_metrics = metrics[name]["test"]
        summary_rows.append({
            "window": window,
            "model": name,
            "auc_pr": test_metrics["auc_pr"],
            "precision_at_0.005": test_metrics["precision_at_0.005"],
            "precision_at_0.01": test_metrics["precision_at_0.01"],
            "precision_at_0.02": test_metrics["precision_at_0.02"],
            "precision_at_0.05": test_metrics["precision_at_0.05"],
            "runtime_seconds": runtime["seconds"],
        })
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({row["model"]: row for row in summary_rows}, indent=2))
    return {"summary": summary_rows, "runtime": runtime, "cohorts": cohorts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--out-dir", default="outputs/ieee_green_focused_improvements_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    runtimes = {}
    cohorts = []
    for window in parse_ints(args.windows):
        result = run_window(args, window)
        summaries.extend(result["summary"])
        runtimes[f"window_{window}"] = result["runtime"]
        for row in result["cohorts"]:
            row["window"] = window
        cohorts.extend(result["cohorts"])
    pd.DataFrame(summaries).to_csv(root / "window_summary.csv", index=False)
    pd.DataFrame(cohorts).to_csv(root / "cohort_metrics.csv", index=False)
    save_json(root / "runtime.json", runtimes)


if __name__ == "__main__":
    main()

