from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.green_risk_field import GreenRiskFieldBuilder, select_by_validation
from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis
from green_fraud_fields.modeling import evaluate, fit_lightgbm, save_json


def base_groups(columns: list[str]) -> dict[str, list[str]]:
    tabular = [
        c for c in columns
        if not c.startswith(("agg__", "cohort_", "known_")) and "__" not in c
    ]
    cold = [c for c in columns if c.startswith((
        "agg__a_seen", "agg__b_seen", "agg__degree", "agg__count_",
        "agg__time_since_a", "agg__time_since_b", "cohort_", "known_",
    ))]
    graph = [c for c in columns if any(k in c for k in (
        "common_neighbors", "jaccard", "adamic_adar", "degree_min", "degree_max",
        "degree_sum", "degree_product",
    ))]
    proximity = [c for c in columns if any(k in c for k in (
        "edge_seen", "count_ab", "time_since_edge", "preferential_attachment",
    ))]
    tcg = list(dict.fromkeys(tabular + cold + graph))
    return {"B": tcg, "BP": list(dict.fromkeys(tcg + proximity))}


def green_columns(schema_names: list[str], delay: int, alpha: int, block: str) -> list[str]:
    marker = f"d{delay}_a{alpha}__{block}_"
    return [column for column in schema_names if marker in column]


def fit_predict(
    features: pd.DataFrame,
    selected: list[str],
    y: np.ndarray,
    split,
    seed: int,
):
    train, valid, test = split
    preprocessor, model = fit_lightgbm(
        features.iloc[train][selected], y[train],
        features.iloc[valid][selected], y[valid], seed,
    )
    valid_prediction = model.predict_proba(
        preprocessor.transform(features.iloc[valid][selected])
    )[:, 1]
    test_prediction = model.predict_proba(
        preprocessor.transform(features.iloc[test][selected])
    )[:, 1]
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({
        "feature": names,
        "importance": model.feature_importances_,
    })
    return valid_prediction, test_prediction, preprocessor, model, importance


def permutation_drop(
    features: pd.DataFrame,
    selected: list[str],
    block: list[str],
    y: np.ndarray,
    rows: slice,
    preprocessor,
    model,
    seed: int,
) -> float:
    original = features.iloc[rows][selected].copy()
    base = model.predict_proba(preprocessor.transform(original))[:, 1]
    shuffled = original.copy()
    rng = np.random.default_rng(seed)
    for column in block:
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    changed = model.predict_proba(preprocessor.transform(shuffled))[:, 1]
    return float(
        average_precision_score(y[rows], base)
        - average_precision_score(y[rows], changed)
    )


def cohort_metrics(
    base_test: pd.DataFrame,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    masks = {
        "all": np.ones(len(base_test), dtype=bool),
        "known_endpoints": base_test["known_endpoints_all_edges"].to_numpy(bool),
        "C00_newedge": base_test["cohort_any_C00_newedge"].to_numpy(bool),
        "identity_present": base_test["identity_present"].to_numpy(bool),
        "identity_missing": ~base_test["identity_present"].to_numpy(bool),
        "C10": base_test["cohort_any_C10"].to_numpy(bool),
        "C11": base_test["cohort_all_C11"].to_numpy(bool),
    }
    output = []
    for cohort, mask in masks.items():
        if mask.sum() < 20 or np.unique(y[mask]).size < 2:
            continue
        for model, prediction in predictions.items():
            output.append({
                "cohort": cohort, "model": model, "rows": int(mask.sum()),
                **evaluate(y[mask], prediction[mask]),
            })
    return output


def paired_bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    alternatives: dict[str, np.ndarray],
    replicates: int = 2000,
    block_size: int = 500,
    seed: int = 23,
) -> dict:
    blocks = [
        np.arange(start, min(start + block_size, len(y)))
        for start in range(0, len(y), block_size)
    ]
    rng = np.random.default_rng(seed)
    result = {}
    def p1(labels, scores):
        count = max(1, int(len(scores) * 0.01))
        return float(np.mean(labels[np.argsort(-scores)[:count]]))
    for name, score in alternatives.items():
        values = []
        for _ in range(replicates):
            index = np.concatenate([
                blocks[i] for i in rng.integers(0, len(blocks), len(blocks))
            ])
            if np.unique(y[index]).size < 2:
                continue
            values.append([
                average_precision_score(y[index], score[index])
                - average_precision_score(y[index], base[index]),
                p1(y[index], score[index]) - p1(y[index], base[index]),
            ])
        array = np.asarray(values)
        result[name] = {
            "replicates": len(array),
            "auc_pr_delta_mean": float(array[:, 0].mean()),
            "auc_pr_delta_ci": np.quantile(array[:, 0], [0.025, 0.975]).tolist(),
            "p1_delta_mean": float(array[:, 1].mean()),
            "p1_delta_ci": np.quantile(array[:, 1], [0.025, 0.975]).tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--out-dir", default="outputs/ieee_green_risk_field_v1")
    parser.add_argument("--base-cache", default="outputs/ieee_tail_50k_r2c100/feature_cache.parquet")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--ego-radius", type=int, default=2)
    parser.add_argument("--max-ego-nodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_ieee_cis(args.data_dir, args.max_rows)
    y = data["isFraud"].to_numpy(int)
    split = chronological_split(len(data))
    base = pd.read_parquet(args.base_cache).iloc[:len(data)].reset_index(drop=True)
    green_path = out / "green_features.parquet"
    if not green_path.exists():
        builder = GreenRiskFieldBuilder(
            lambda_=args.lambda_,
            ego_radius=args.ego_radius,
            max_ego_nodes=args.max_ego_nodes,
        )
        builder.write_parquet(data, str(green_path))
        ego_mean = float(np.mean(builder.ego_sizes)) if builder.ego_sizes else 0.0
    else:
        ego_mean = np.nan
    parquet = pq.ParquetFile(green_path)
    schema_names = parquet.schema.names
    groups = base_groups(list(base.columns))
    metrics: dict = {}
    selected_params: dict = {"lambda": args.lambda_, "selection": {}}
    cohort_rows: list[dict] = []
    importance_rows = []
    permutation = {}
    test_predictions: dict[str, np.ndarray] = {}
    for family, baseline_columns in groups.items():
        baseline_valid, baseline_test, _, _, baseline_importance = fit_predict(
            base, baseline_columns, y, split, args.seed
        )
        metrics[f"{family}0"] = {
            "validation": evaluate(y[split[1]], baseline_valid),
            "test": evaluate(y[split[2]], baseline_test),
        }
        test_predictions[f"{family}0"] = baseline_test
        baseline_importance.assign(model=f"{family}0", delay=-1, alpha=-1)
        for delay in (0, 1, 3, 7, 14, 30, 60):
            alpha_trials = []
            for alpha in (5, 20, 100):
                hcols = green_columns(schema_names, delay, alpha, "H")
                scols = green_columns(schema_names, delay, alpha, "S")
                green = pd.read_parquet(green_path, columns=hcols + scols)
                frame = pd.concat([base, green], axis=1)
                valid_prediction, test_prediction, pre, model, importance = fit_predict(
                    frame, baseline_columns + hcols + scols, y, split, args.seed
                )
                alpha_trials.append((
                    evaluate(y[split[1]], valid_prediction)["auc_pr"],
                    alpha, frame, hcols, scols, valid_prediction, test_prediction,
                    pre, model, importance,
                ))
            chosen = select_by_validation(alpha_trials)
            _, alpha, frame, hcols, scols, b2_valid, b2_test, b2_pre, b2_model, b2_imp = chosen
            trcols = green_columns(schema_names, delay, alpha, "TR")
            if trcols:
                transfer = pd.read_parquet(green_path, columns=trcols)
                frame = pd.concat([frame, transfer], axis=1)
            b1_valid, b1_test, _, _, b1_imp = fit_predict(
                frame, baseline_columns + hcols, y, split, args.seed
            )
            b3_valid, b3_test, b3_pre, b3_model, b3_imp = fit_predict(
                frame, baseline_columns + hcols + scols + trcols, y, split, args.seed
            )
            names = {
                f"{family}1_d{delay}": (b1_valid, b1_test),
                f"{family}2_d{delay}": (b2_valid, b2_test),
                f"{family}3_d{delay}": (b3_valid, b3_test),
            }
            for name, (valid_prediction, test_prediction) in names.items():
                metrics[name] = {
                    "validation": evaluate(y[split[1]], valid_prediction),
                    "test": evaluate(y[split[2]], test_prediction),
                }
                test_predictions[name] = test_prediction
            selected_params["selection"][f"{family}_d{delay}"] = {
                "alpha": alpha,
                "alpha_validation_auc_pr": {
                    str(item[1]): item[0] for item in alpha_trials
                },
            }
            for model_name, table in (
                (f"{family}1_d{delay}", b1_imp),
                (f"{family}2_d{delay}", b2_imp),
                (f"{family}3_d{delay}", b3_imp),
            ):
                table = table.copy()
                table["model"] = model_name
                table["delay"] = delay
                table["alpha"] = alpha
                importance_rows.append(table)
            permutation[f"{family}_d{delay}"] = {
                "S_validation_ap_drop": permutation_drop(
                    frame, baseline_columns + hcols + scols, scols, y, split[1],
                    b2_pre, b2_model, args.seed,
                ),
                "S_test_ap_drop": permutation_drop(
                    frame, baseline_columns + hcols + scols, scols, y, split[2],
                    b2_pre, b2_model, args.seed,
                ),
                "TR_validation_ap_drop": permutation_drop(
                    frame, baseline_columns + hcols + scols + trcols, trcols, y, split[1],
                    b3_pre, b3_model, args.seed,
                ),
                "TR_test_ap_drop": permutation_drop(
                    frame, baseline_columns + hcols + scols + trcols, trcols, y, split[2],
                    b3_pre, b3_model, args.seed,
                ),
            }
    test_base = base.iloc[split[2]].reset_index(drop=True)
    cohort_rows = cohort_metrics(test_base, y[split[2]], test_predictions)
    bootstrap = {}
    for family in groups:
        for delay in (0, 1, 3, 7, 14, 30, 60):
            bootstrap[f"{family}_d{delay}"] = paired_bootstrap(
                y[split[2]],
                test_predictions[f"{family}1_d{delay}"],
                {
                    "B2_minus_B1": test_predictions[f"{family}2_d{delay}"],
                    "B3_minus_B1": test_predictions[f"{family}3_d{delay}"],
                },
            )
    test_data = data.iloc[split[2]].reset_index(drop=True)
    prediction_frame = pd.DataFrame({
        "TransactionID": test_data["TransactionID"],
        "isFraud": y[split[2]],
        "identity_present": test_base["identity_present"],
        "known_endpoints": test_base["known_endpoints_all_edges"],
        "C00_newedge": test_base["cohort_any_C00_newedge"],
        "C10": test_base["cohort_any_C10"],
        "C11": test_base["cohort_all_C11"],
        "split_window": "window_0_test",
        **{f"score_{name}": score for name, score in test_predictions.items()},
    })
    prediction_frame.to_parquet(out / "predictions_test.parquet", index=False)
    save_json(out / "metrics.json", metrics)
    save_json(out / "cohort_metrics.json", {"rows": cohort_rows})
    save_json(out / "bootstrap.json", bootstrap)
    save_json(out / "selected_hyperparams.json", selected_params | {
        "permutation_ap_drop": permutation
    })
    if importance_rows:
        pd.concat(importance_rows, ignore_index=True).to_csv(
            out / "feature_importance.csv", index=False
        )
    save_json(out / "runtime.json", {
        "total_seconds": time.time() - started,
        "mean_ego_size": ego_mean,
        "rows": len(data),
        "lambda": args.lambda_,
        "ego_radius": args.ego_radius,
        "max_ego_nodes": args.max_ego_nodes,
        "random_forest": "not used",
    })
    print(json.dumps({
        key: value["test"] for key, value in metrics.items()
        if key.startswith("B") and ("_d" in key or key.endswith("0"))
    }, indent=2))


if __name__ == "__main__":
    main()

