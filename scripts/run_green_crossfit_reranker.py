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
from green_fraud_fields.modeling import evaluate, make_preprocessor, save_json
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached, stable_model_key
from run_green_adaptive_theory import add_adaptive_shrinkage
from run_green_focused_improvements import tuned_soft_mixture
from run_green_reranker_upgrade import logistic_tail_reranker, save_predictions
from run_green_risk_field import base_groups, cohort_metrics, green_columns
from run_green_risk_tail import rerank


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def safe_logit(score: np.ndarray) -> np.ndarray:
    return logit(np.clip(score, 1e-6, 1 - 1e-6))


def chronological_oof_slices(train: slice, folds: int, warmup_fraction: float) -> list[slice]:
    start = train.start or 0
    stop = train.stop
    warmup_end = start + int((stop - start) * warmup_fraction)
    points = np.linspace(warmup_end, stop, folds + 1, dtype=int)
    return [slice(int(points[i]), int(points[i + 1])) for i in range(folds) if points[i + 1] > points[i]]


def fit_fixed_chrono_lgbm(
    frame: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if len(train_idx) < 200 or np.unique(y[train_idx]).size < 2:
        prior = float(np.mean(y[train_idx])) if len(train_idx) else float(np.mean(y))
        return np.full(len(pred_idx), prior), {"fallback": "constant_prior", "prior": prior}

    inner_cut = max(1, int(len(train_idx) * 0.8))
    inner_train = train_idx[:inner_cut]
    inner_valid = train_idx[inner_cut:]
    if len(inner_valid) < 100 or np.unique(y[inner_valid]).size < 2:
        inner_train = train_idx
        inner_valid = train_idx

    preprocessor = make_preprocessor(frame.iloc[inner_train][columns])
    x_train = preprocessor.fit_transform(frame.iloc[inner_train][columns])
    x_valid = preprocessor.transform(frame.iloc[inner_valid][columns])
    x_pred = preprocessor.transform(frame.iloc[pred_idx][columns])
    positives = max(float(y[inner_train].sum()), 1.0)
    scale = (len(inner_train) - positives) / positives
    model = LGBMClassifier(
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        reg_lambda=3.0,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y[inner_train],
        eval_set=[(x_valid, y[inner_valid])],
        eval_metric="average_precision",
        callbacks=[early_stopping(25, verbose=False)],
    )
    return model.predict_proba(x_pred)[:, 1], {
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "train_rows": int(len(inner_train)),
        "valid_rows": int(len(inner_valid)),
    }


def oof_base_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    train: slice,
    model_columns: dict[str, list[str]],
    folds: int,
    warmup_fraction: float,
    seed: int,
    cache_dir: Path,
    force: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fold_slices = chronological_oof_slices(train, folds, warmup_fraction)
    oof_index = np.concatenate([np.arange(s.start, s.stop) for s in fold_slices])
    predictions = {name: np.full(len(oof_index), np.nan, dtype=np.float64) for name in model_columns}
    details: dict = {
        "folds": [{"start": int(s.start), "stop": int(s.stop)} for s in fold_slices],
        "warmup_fraction": warmup_fraction,
        "models": {},
        "test_used_for_selection": False,
    }
    offset_by_start = {}
    cursor = 0
    for fold in fold_slices:
        length = fold.stop - fold.start
        offset_by_start[fold.start] = slice(cursor, cursor + length)
        cursor += length

    for name, columns in model_columns.items():
        key = stable_model_key(f"oof_{name}_f{folds}_w{warmup_fraction}", columns, seed)
        path = cache_dir / f"{name}.{key}.npz"
        meta_path = cache_dir / f"{name}.{key}.json"
        if path.exists() and meta_path.exists() and not force:
            predictions[name] = np.load(path)["prediction"]
            details["models"][name] = json.loads(meta_path.read_text(encoding="utf-8"))
            details["models"][name]["cached"] = True
            continue
        model_detail = {"cached": False, "folds": []}
        for fold_id, fold in enumerate(fold_slices):
            pred_idx = np.arange(fold.start, fold.stop)
            train_idx = np.arange(train.start or 0, fold.start)
            pred, fold_detail = fit_fixed_chrono_lgbm(frame, y, columns, train_idx, pred_idx, seed + fold_id)
            predictions[name][offset_by_start[fold.start]] = pred
            model_detail["folds"].append({"fold": fold_id, **fold_detail})
        np.savez_compressed(path, prediction=predictions[name])
        save_json(meta_path, model_detail)
        details["models"][name] = model_detail
    return oof_index, predictions, details


def oof_meta_matrices(
    frame: pd.DataFrame,
    oof_index: np.ndarray,
    valid: slice,
    test: slice,
    oof_predictions: dict[str, np.ndarray],
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_oof = frame.iloc[oof_index][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_valid = frame.iloc[valid][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_test = frame.iloc[test][summary_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    summary_oof = imputer.fit_transform(raw_oof)
    summary_valid = imputer.transform(raw_valid)
    summary_test = imputer.transform(raw_test)
    meta_oof = np.column_stack([*[safe_logit(oof_predictions[name]) for name in model_names], summary_oof])
    meta_valid = np.column_stack([*[safe_logit(valid_predictions[name]) for name in model_names], summary_valid])
    meta_test = np.column_stack([*[safe_logit(test_predictions[name]) for name in model_names], summary_test])
    return meta_oof, meta_valid, meta_test


def crossfit_logistic_tail(
    frame: pd.DataFrame,
    y: np.ndarray,
    oof_index: np.ndarray,
    valid: slice,
    test: slice,
    oof_predictions: dict[str, np.ndarray],
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
    fractions: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta_oof, meta_valid, meta_test = oof_meta_matrices(
        frame, oof_index, valid, test, oof_predictions, valid_predictions, test_predictions, model_names, summary_cols
    )
    y_oof = y[oof_index]
    y_valid = y[valid]
    best = None
    trials = []
    for fraction in fractions:
        train_mask = oof_predictions["M3"] >= np.quantile(oof_predictions["M3"], 1 - fraction)
        if np.unique(y_oof[train_mask]).size < 2:
            continue
        model = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0)
        model.fit(meta_oof[train_mask], y_oof[train_mask])
        valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
        valid_probability = model.predict_proba(meta_valid)[:, 1]
        valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
        metrics = evaluate(y_valid, valid_score)
        key = (metrics["precision_at_0.01"], metrics["precision_at_0.005"], metrics["auc_pr"])
        trials.append({"fraction": fraction, "selection_key": list(key), "validation": metrics})
        if best is None or key > best[0]:
            best = (key, fraction, model)
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in OOF tail candidates",
            "test_used_for_selection": False,
        }
    _, fraction, model = best
    valid_probability = model.predict_proba(meta_valid)[:, 1]
    test_probability = model.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    test_mask = test_predictions["M3"] >= np.quantile(test_predictions["M3"], 1 - fraction)
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "OOF-trained logistic reranker, validation-selected candidate fraction by P@1%, P@0.5%, AUC-PR",
        "candidate_fraction": fraction,
        "selection_key": list(best[0]),
        "trials": trials,
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


def crossfit_lgbm_tail(
    frame: pd.DataFrame,
    y: np.ndarray,
    oof_index: np.ndarray,
    valid: slice,
    test: slice,
    oof_predictions: dict[str, np.ndarray],
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
    fractions: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta_oof, meta_valid, meta_test = oof_meta_matrices(
        frame, oof_index, valid, test, oof_predictions, valid_predictions, test_predictions, model_names, summary_cols
    )
    y_oof = y[oof_index]
    y_valid = y[valid]
    params_grid = [
        {"num_leaves": 7, "min_child_samples": 20, "reg_lambda": 5.0},
        {"num_leaves": 15, "min_child_samples": 30, "reg_lambda": 3.0},
    ]
    best = None
    trials = []
    for fraction in fractions:
        train_mask = oof_predictions["M3"] >= np.quantile(oof_predictions["M3"], 1 - fraction)
        if np.unique(y_oof[train_mask]).size < 2:
            continue
        positives = max(float(y_oof[train_mask].sum()), 1.0)
        scale = (float(train_mask.sum()) - positives) / positives
        for idx, params in enumerate(params_grid):
            model = LGBMClassifier(
                n_estimators=180,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale,
                random_state=seed + idx,
                n_jobs=-1,
                verbosity=-1,
                **params,
            )
            model.fit(
                meta_oof[train_mask],
                y_oof[train_mask],
                eval_set=[(meta_valid, y_valid)],
                eval_metric="average_precision",
                callbacks=[early_stopping(25, verbose=False)],
            )
            valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
            valid_probability = model.predict_proba(meta_valid)[:, 1]
            valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
            metrics = evaluate(y_valid, valid_score)
            key = (metrics["precision_at_0.01"], metrics["precision_at_0.005"], metrics["auc_pr"])
            trial = {
                "fraction": fraction,
                "params": params,
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
                "selection_key": list(key),
                "validation": metrics,
            }
            trials.append(trial)
            if best is None or key > best[0]:
                best = (key, fraction, params, int(model.best_iteration_ or model.n_estimators))
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in OOF tail candidates",
            "test_used_for_selection": False,
        }
    _, fraction, params, best_iteration = best
    train_mask = oof_predictions["M3"] >= np.quantile(oof_predictions["M3"], 1 - fraction)
    positives = max(float(y_oof[train_mask].sum()), 1.0)
    scale = (float(train_mask.sum()) - positives) / positives
    model = LGBMClassifier(
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
    model.fit(meta_oof[train_mask], y_oof[train_mask])
    valid_probability = model.predict_proba(meta_valid)[:, 1]
    test_probability = model.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    test_mask = test_predictions["M3"] >= np.quantile(test_predictions["M3"], 1 - fraction)
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "OOF-trained LightGBM reranker, validation-selected candidate fraction by P@1%, P@0.5%, AUC-PR",
        "candidate_fraction": fraction,
        "params": params,
        "best_iteration": best_iteration,
        "selection_key": list(best[0]),
        "trials": trials,
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


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
    model_cols = {
        "M3": baseline_cols,
        "H_S": baseline_cols + hcols + scols,
        "adaptive_shrinkage": baseline_cols + hcols + sg_cols,
        "precision_green_count": baseline_cols + hcols + [c for c in sd_cols if "SD_count" in c],
        "precision_green_sqrt": baseline_cols + hcols + [c for c in sd_cols if "SD_sqrt" in c],
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
    timer.mark("fit_or_load_full_base")

    soft_valid, soft_test, soft_selection = tuned_soft_mixture(
        y[valid], valid_predictions, test_predictions,
        ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt"],
    )
    valid_predictions["adaptive_soft_current"] = soft_valid
    test_predictions["adaptive_soft_current"] = soft_test
    selection["adaptive_soft_current"] = soft_selection
    timer.mark("soft_mixture")

    oof_index, oof_predictions, oof_details = oof_base_predictions(
        frame,
        y,
        train,
        model_cols,
        args.oof_folds,
        args.oof_warmup_fraction,
        args.seed,
        out / "oof_prediction_cache",
        args.force_oof,
    )
    timer.mark("oof_base_predictions")

    summary_cols = [c for c in sg_cols + sd_cols if c.startswith("agg__") or "all_" in c]
    model_names = ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt"]
    fractions = parse_floats(args.tail_fractions)

    xlog_valid, xlog_test, xlog_selection = crossfit_logistic_tail(
        frame, y, oof_index, valid, test, oof_predictions, valid_predictions, test_predictions, model_names, summary_cols, fractions
    )
    valid_predictions["crossfit_logistic_tail"] = xlog_valid
    test_predictions["crossfit_logistic_tail"] = xlog_test
    selection["crossfit_logistic_tail"] = xlog_selection
    timer.mark("crossfit_logistic_tail")

    xlgb_valid, xlgb_test, xlgb_selection = crossfit_lgbm_tail(
        frame, y, oof_index, valid, test, oof_predictions, valid_predictions, test_predictions, model_names, summary_cols, fractions, args.seed
    )
    valid_predictions["crossfit_lgbm_tail"] = xlgb_valid
    test_predictions["crossfit_lgbm_tail"] = xlgb_test
    selection["crossfit_lgbm_tail"] = xlgb_selection
    timer.mark("crossfit_lgbm_tail")

    # Reproduce the accepted split-validation logistic tail in the same output
    # folder so reference comparisons are easy to inspect.
    ref_valid, ref_test, ref_selection = logistic_tail_reranker(
        frame, y, valid, test, valid_predictions, test_predictions,
        ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt", "adaptive_soft_current"],
        summary_cols,
        fractions,
    )
    valid_predictions["split_valid_logistic_tail"] = ref_valid
    test_predictions["split_valid_logistic_tail"] = ref_test
    selection["split_valid_logistic_tail"] = ref_selection
    timer.mark("split_valid_reference_tail")

    blend_valid, blend_test, blend_selection = tuned_soft_mixture(
        y[valid],
        valid_predictions,
        test_predictions,
        ["M3", "adaptive_soft_current", "crossfit_logistic_tail", "crossfit_lgbm_tail", "split_valid_logistic_tail"],
    )
    valid_predictions["crossfit_tail_blend"] = blend_valid
    test_predictions["crossfit_tail_blend"] = blend_test
    selection["crossfit_tail_blend"] = blend_selection
    selection["_oof"] = oof_details
    timer.mark("blend")

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
        "oof_rows": int(len(oof_index)),
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

    reference_path = reference_dir / "window_summary.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        reference = reference[reference["model"] == "adaptive_two_stage"].set_index("window")
    else:
        reference = summary[summary["model"] == "split_valid_logistic_tail"].set_index("window")
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

    ref_cohort_path = reference_dir / "cohort_metrics.csv"
    if ref_cohort_path.exists():
        ref_cohorts = pd.read_csv(ref_cohort_path)
        ref_cohorts = ref_cohorts[ref_cohorts["model"] == "adaptive_two_stage"]
    else:
        ref_cohorts = cohort_frame[cohort_frame["model"] == "split_valid_logistic_tail"]
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
    parser.add_argument("--out-dir", default="outputs/ieee_green_crossfit_reranker_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--tail-fractions", default="0.025,0.05,0.10,0.20,0.30")
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-warmup-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--force-oof", action="store_true")
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

