from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis, tabular_features
from green_fraud_fields.modeling import evaluate, make_preprocessor, save_json
from green_fraud_fields.temporal_features import CausalFeatureBuilder


MODEL_NAMES = ["M3", "M3_F", "M3_T", "M3_V", "M3_FT", "M6"]


def feature_columns(columns: list[str], model: str) -> list[str]:
    tabular = [
        c for c in columns
        if not c.startswith(("agg__", "cohort_", "known_")) and "__" not in c
    ]
    cold = [c for c in columns if c.startswith((
        "agg__a_seen", "agg__b_seen", "agg__edge_seen", "agg__degree",
        "agg__count_", "agg__time_since", "cohort_", "known_",
    ))]
    graph = [c for c in columns if any(k in c for k in (
        "common_neighbors", "jaccard", "adamic_adar", "degree_min", "degree_max",
        "degree_sum", "degree_product",
    ))]
    temporal = [c for c in columns if any(k in c for k in (
        "edge_seen", "count_ab", "time_since_edge", "preferential_attachment",
    ))]
    laplacian = [c for c in columns if any(k in c for k in (
        "G_aa", "G_bb", "G_ab", "R_ab", "logdet_edge", "fragility_",
        "cross_ratio", "resistance_additive_gap",
    ))]
    tension = [c for c in columns if any(k in c for k in (
        "t_edge_", "t_endpoint_", "delta_log_amt_mean",
    ))]
    velocity = [c for c in columns if "count_velocity" in c]
    baseline = tabular + cold + graph + temporal
    mapping = {
        "M3": baseline,
        "M3_F": baseline + laplacian,
        "M3_T": baseline + tension,
        "M3_V": baseline + velocity,
        "M3_FT": baseline + laplacian + tension,
        "M6": baseline + laplacian + tension + velocity,
    }
    return list(dict.fromkeys(mapping[model]))


def tail_objective(y: np.ndarray, prediction: np.ndarray) -> tuple[float, ...]:
    metrics = evaluate(y, prediction)
    return (
        metrics["precision_at_0.01"],
        metrics["precision_at_0.005"],
        metrics["precision_at_0.02"],
        metrics["auc_pr"],
    )


def fit_tail_selected(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    preprocessor = make_preprocessor(x_train)
    train_matrix = preprocessor.fit_transform(x_train)
    valid_matrix = preprocessor.transform(x_valid)
    test_matrix = preprocessor.transform(x_test)
    positives = max(float(y_train.sum()), 1.0)
    scale = (len(y_train) - positives) / positives
    candidates = [
        {"num_leaves": 15, "min_child_samples": 40, "reg_lambda": 2.0},
        {"num_leaves": 31, "min_child_samples": 20, "reg_lambda": 1.0},
        {"num_leaves": 63, "min_child_samples": 30, "reg_lambda": 3.0},
        {"num_leaves": 31, "min_child_samples": 80, "reg_lambda": 5.0},
    ]
    best = None
    trials = []
    candidate_predictions = []
    for index, params in enumerate(candidates):
        model = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale,
            random_state=seed + index,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(
            train_matrix,
            y_train,
            eval_set=[(valid_matrix, y_valid)],
            eval_metric="average_precision",
            callbacks=[early_stopping(50, verbose=False)],
        )
        valid_prediction = model.predict_proba(valid_matrix)[:, 1]
        test_prediction = model.predict_proba(test_matrix)[:, 1]
        objective = tail_objective(y_valid, valid_prediction)
        trial = {
            **params,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "validation": evaluate(y_valid, valid_prediction),
        }
        trials.append(trial)
        candidate_predictions.append({
            "params": {**params, "best_iteration": trial["best_iteration"]},
            "valid": valid_prediction,
            "test": test_prediction,
        })
        if best is None or objective > best[0]:
            best = (objective, model, valid_prediction, trial)
    assert best is not None
    test_prediction = best[1].predict_proba(test_matrix)[:, 1]
    return best[2], best[1].predict_proba(test_matrix)[:, 1], {
        "selected": best[3],
        "trials": trials,
    }, candidate_predictions


def tune_logit_mixture(
    y_valid: np.ndarray,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    epsilon = 1e-6
    base_valid = logit(np.clip(valid_predictions["M3"], epsilon, 1 - epsilon))
    base_test = logit(np.clip(test_predictions["M3"], epsilon, 1 - epsilon))
    deltas_valid = {
        key: logit(np.clip(valid_predictions[key], epsilon, 1 - epsilon)) - base_valid
        for key in ("M3_F", "M3_T", "M3_V")
    }
    deltas_test = {
        key: logit(np.clip(test_predictions[key], epsilon, 1 - epsilon)) - base_test
        for key in ("M3_F", "M3_T", "M3_V")
    }
    weights = (-0.5, 0.0, 0.25, 0.5, 1.0, 1.5)
    best = None
    for alpha, beta, gamma in product(weights, repeat=3):
        valid_score = expit(
            base_valid
            + alpha * deltas_valid["M3_F"]
            + beta * deltas_valid["M3_T"]
            + gamma * deltas_valid["M3_V"]
        )
        objective = tail_objective(y_valid, valid_score)
        if best is None or objective > best[0]:
            best = (objective, alpha, beta, gamma, valid_score)
    assert best is not None
    test_score = expit(
        base_test
        + best[1] * deltas_test["M3_F"]
        + best[2] * deltas_test["M3_T"]
        + best[3] * deltas_test["M3_V"]
    )
    return best[4], test_score, {
        "alpha_F": best[1],
        "beta_tension": best[2],
        "gamma_velocity": best[3],
        "validation": evaluate(y_valid, best[4]),
    }


def gated_predictions(
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_valid: np.ndarray,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_masks = {
        "C00_newedge": valid_frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": (
            valid_frame["known_endpoints_all_edges"].to_numpy(bool)
            & ~valid_frame["cohort_any_C00_newedge"].to_numpy(bool)
        ),
    }
    valid_masks["other"] = ~(valid_masks["C00_newedge"] | valid_masks["known_endpoints"])
    test_masks = {
        "C00_newedge": test_frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": (
            test_frame["known_endpoints_all_edges"].to_numpy(bool)
            & ~test_frame["cohort_any_C00_newedge"].to_numpy(bool)
        ),
    }
    test_masks["other"] = ~(test_masks["C00_newedge"] | test_masks["known_endpoints"])
    chosen: dict[str, str] = {}
    valid_score = np.zeros(len(valid_frame))
    test_score = np.zeros(len(test_frame))
    for cohort, mask in valid_masks.items():
        candidates = []
        for model in MODEL_NAMES:
            if mask.sum() >= 20 and np.unique(y_valid[mask]).size == 2:
                objective = tail_objective(y_valid[mask], valid_predictions[model][mask])
            else:
                objective = tail_objective(y_valid, valid_predictions[model])
            candidates.append((objective, model))
        model = max(candidates)[1]
        chosen[cohort] = model
        valid_score[mask] = valid_predictions[model][mask]
        test_score[test_masks[cohort]] = test_predictions[model][test_masks[cohort]]
    return valid_score, test_score, {
        "chosen_models": chosen,
        "validation": evaluate(y_valid, valid_score),
    }


def robust_candidate_gate(
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_valid: np.ndarray,
    candidate_pool: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_masks = {
        "C00_newedge": valid_frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": (
            valid_frame["known_endpoints_all_edges"].to_numpy(bool)
            & ~valid_frame["cohort_any_C00_newedge"].to_numpy(bool)
        ),
    }
    valid_masks["other"] = ~(valid_masks["C00_newedge"] | valid_masks["known_endpoints"])
    test_masks = {
        "C00_newedge": test_frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": (
            test_frame["known_endpoints_all_edges"].to_numpy(bool)
            & ~test_frame["cohort_any_C00_newedge"].to_numpy(bool)
        ),
    }
    test_masks["other"] = ~(test_masks["C00_newedge"] | test_masks["known_endpoints"])
    valid_score = np.zeros(len(valid_frame))
    test_score = np.zeros(len(test_frame))
    choices = {}
    for cohort, mask in valid_masks.items():
        best = None
        for candidate in candidate_pool:
            global_metrics = evaluate(y_valid, candidate["valid"])
            cohort_metrics = evaluate(y_valid[mask], candidate["valid"][mask])
            objective = (
                0.65 * cohort_metrics["precision_at_0.01"]
                + 0.35 * global_metrics["precision_at_0.01"],
                0.65 * cohort_metrics["precision_at_0.005"]
                + 0.35 * global_metrics["precision_at_0.005"],
                cohort_metrics["auc_pr"],
            )
            if best is None or objective > best[0]:
                best = (objective, candidate)
        assert best is not None
        chosen = best[1]
        valid_score[mask] = chosen["valid"][mask]
        test_score[test_masks[cohort]] = chosen["test"][test_masks[cohort]]
        choices[cohort] = {
            "model": chosen["model"],
            "params": chosen["params"],
        }
    return valid_score, test_score, {
        "chosen_candidates": choices,
        "validation": evaluate(y_valid, valid_score),
        "selection": "65% cohort P@k + 35% global P@k shrinkage",
    }


def rerank_score(base: np.ndarray, reranker: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    score = np.asarray(base, dtype=float).copy()
    if not candidate_mask.any():
        return score
    boundary = np.max(score[~candidate_mask]) if (~candidate_mask).any() else 0.0
    ranks = pd.Series(reranker[candidate_mask]).rank(method="average", pct=True).to_numpy()
    score[candidate_mask] = boundary + 1.0 + ranks
    maximum = float(np.max(score))
    return score / maximum if maximum > 0 else score


def tune_two_stage(
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_valid: np.ndarray,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    split = len(valid_frame) // 2
    meta_features_valid = np.column_stack([
        logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6))
        for name in MODEL_NAMES
    ] + [
        valid_frame["cohort_any_C00_newedge"].to_numpy(),
        valid_frame["known_endpoints_all_edges"].to_numpy(),
    ])
    meta_features_test = np.column_stack([
        logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6))
        for name in MODEL_NAMES
    ] + [
        test_frame["cohort_any_C00_newedge"].to_numpy(),
        test_frame["known_endpoints_all_edges"].to_numpy(),
    ])
    best = None
    for fraction in (0.05, 0.10, 0.20):
        first_base = valid_predictions["M3"][:split]
        cutoff = np.quantile(first_base, 1 - fraction)
        train_mask = first_base >= cutoff
        if np.unique(y_valid[:split][train_mask]).size < 2:
            continue
        reranker = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=0.5, random_state=0
        )
        reranker.fit(meta_features_valid[:split][train_mask], y_valid[:split][train_mask])
        second_base = valid_predictions["M3"][split:]
        second_cutoff = np.quantile(second_base, 1 - fraction)
        second_mask = second_base >= second_cutoff
        second_probability = reranker.predict_proba(meta_features_valid[split:])[:, 1]
        second_score = rerank_score(second_base, second_probability, second_mask)
        objective = tail_objective(y_valid[split:], second_score)
        if best is None or objective > best[0]:
            best = (objective, fraction, reranker, second_score)
    if best is None:
        return (
            valid_predictions["M3"].copy(),
            test_predictions["M3"].copy(),
            {
                "fallback": "M3",
                "reason": "insufficient positives in validation candidate sets",
            },
        )
    fraction, reranker = best[1], best[2]
    valid_cutoff = np.quantile(valid_predictions["M3"], 1 - fraction)
    valid_mask = valid_predictions["M3"] >= valid_cutoff
    valid_probability = reranker.predict_proba(meta_features_valid)[:, 1]
    valid_score = rerank_score(valid_predictions["M3"], valid_probability, valid_mask)
    test_cutoff = np.quantile(test_predictions["M3"], 1 - fraction)
    test_mask = test_predictions["M3"] >= test_cutoff
    test_probability = reranker.predict_proba(meta_features_test)[:, 1]
    test_score = rerank_score(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "candidate_fraction": fraction,
        "selection_half_validation": evaluate(y_valid[split:], best[3]),
        "full_validation_diagnostic": evaluate(y_valid, valid_score),
    }


def cohort_table(
    frame: pd.DataFrame,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "C00_newedge": frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": frame["known_endpoints_all_edges"].to_numpy(bool),
        "identity_present": frame["identity_present"].to_numpy(bool),
    }
    rows = []
    for cohort, mask in masks.items():
        if mask.sum() < 20 or np.unique(y[mask]).size < 2:
            continue
        for name, prediction in predictions.items():
            rows.append({
                "cohort": cohort, "model": name, "rows": int(mask.sum()),
                **evaluate(y[mask], prediction[mask]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--out-dir", default="outputs/ieee_tail_50k")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--ego-radius", type=int, default=2)
    parser.add_argument("--max-ego-nodes", type=int, default=100)
    parser.add_argument("--feature-cache")
    args = parser.parse_args()
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.feature_cache) if args.feature_cache else out_dir / "feature_cache.parquet"
    data = load_ieee_cis(args.data_dir, args.max_rows)
    if cache.exists():
        features = pd.read_parquet(cache)
    else:
        tabular = tabular_features(data)
        causal = CausalFeatureBuilder(
            lambda_=args.lambda_,
            ego_radius=args.ego_radius,
            max_ego_nodes=args.max_ego_nodes,
        ).transform(data)
        features = pd.concat([tabular, causal], axis=1)
        features = features.loc[:, ~features.columns.duplicated()]
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache, index=False)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    valid_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    selection: dict[str, dict] = {}
    candidate_pool: list[dict] = []
    for model_name in MODEL_NAMES:
        selected = feature_columns(list(features.columns), model_name)
        valid_prediction, test_prediction, details, candidates = fit_tail_selected(
            features.iloc[train][selected], y[train],
            features.iloc[valid][selected], y[valid],
            features.iloc[test][selected], args.seed,
        )
        valid_predictions[model_name] = valid_prediction
        test_predictions[model_name] = test_prediction
        selection[model_name] = details
        for candidate in candidates:
            candidate_pool.append({"model": model_name, **candidate})
    valid_frame = features.iloc[valid].reset_index(drop=True)
    test_frame = features.iloc[test].reset_index(drop=True)
    for name, function in (
        ("TAIL_MIX", tune_logit_mixture),
        ("COHORT_GATE", lambda yv, vp, tp: gated_predictions(
            valid_frame, test_frame, yv, vp, tp
        )),
        ("ROBUST_GATE", lambda yv, vp, tp: robust_candidate_gate(
            valid_frame, test_frame, yv, candidate_pool
        )),
        ("TWO_STAGE", lambda yv, vp, tp: tune_two_stage(
            valid_frame, test_frame, yv, vp, tp
        )),
    ):
        valid_score, test_score, details = function(y[valid], valid_predictions, test_predictions)
        valid_predictions[name] = valid_score
        test_predictions[name] = test_score
        selection[name] = details
    metrics = {
        name: {
            "validation": evaluate(y[valid], valid_predictions[name]),
            "test": evaluate(y[test], test_predictions[name]),
        }
        for name in test_predictions
    }
    cohort_table(test_frame, y[test], test_predictions).to_csv(
        out_dir / "cohort_metrics.csv", index=False
    )
    prediction_frame = pd.DataFrame({
        "TransactionID": data.iloc[test]["TransactionID"].to_numpy(),
        "TransactionDT": data.iloc[test]["TransactionDT"].to_numpy(),
        "isFraud": y[test],
        **{f"prediction_{name}": value for name, value in test_predictions.items()},
    })
    prediction_frame.to_parquet(out_dir / "predictions.parquet", index=False)
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "selection.json", selection)
    save_json(out_dir / "config.json", vars(args) | {
        "rows": len(data),
        "runtime_seconds": time.time() - started,
        "random_forest": "not used",
        "test_used_for_selection": False,
    })
    print(json.dumps({name: value["test"] for name, value in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
