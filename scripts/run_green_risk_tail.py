from __future__ import annotations

import json
import argparse
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

from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis, tabular_features
from green_fraud_fields.modeling import evaluate, save_json
from green_fraud_fields.green_risk_field import GreenRiskFieldBuilder
from green_fraud_fields.temporal_features import CausalFeatureBuilder
from run_ieee_tail_specialized import fit_tail_selected, tail_objective
from run_green_risk_field import base_groups, cohort_metrics, green_columns, paired_bootstrap


def rerank(base: np.ndarray, second: np.ndarray, mask: np.ndarray) -> np.ndarray:
    score = base.copy()
    boundary = np.max(score[~mask]) if (~mask).any() else 0.0
    ranks = pd.Series(second[mask]).rank(method="average", pct=True).to_numpy()
    score[mask] = boundary + 1.0 + ranks
    return score / np.max(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/ieee_green_risk_field_v1")
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    offset = args.window * 50000
    full = load_ieee_cis(args.data_dir, offset + 50000)
    data = full.iloc[offset:offset + 50000].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    base_path = out / "base_features.parquet"
    if args.window == 0 and not base_path.exists():
        existing = Path("outputs/ieee_tail_50k_r2c100/feature_cache.parquet")
        if existing.exists():
            base = pd.read_parquet(existing).iloc[:len(data)].reset_index(drop=True)
            base.to_parquet(base_path, index=False)
        else:
            base = None
    else:
        base = pd.read_parquet(base_path) if base_path.exists() else None
    if base is None:
        base = pd.concat([
            tabular_features(data),
            CausalFeatureBuilder(
                compute_laplacian=False, ego_radius=2, max_ego_nodes=100
            ).transform(data),
        ], axis=1)
        base = base.loc[:, ~base.columns.duplicated()]
        base.to_parquet(base_path, index=False)
    green_path = out / "green_features.parquet"
    feature_runtime = None
    if args.force_features and green_path.exists():
        green_path.unlink()
    if not green_path.exists():
        builder = GreenRiskFieldBuilder(
            delays_days=(1,), alphas=(5,), lambda_=0.1,
            ego_radius=2, max_ego_nodes=100,
        )
        feature_started = time.time()
        builder.write_parquet(data, str(green_path))
        feature_runtime = {
            "feature_seconds": time.time() - feature_started,
            **builder.runtime_summary(),
        }
    schema = pq.ParquetFile(green_path).schema.names
    delay, alpha = 1, 5
    hcols = green_columns(schema, delay, alpha, "H")
    scols = green_columns(schema, delay, alpha, "S")
    trcols = green_columns(schema, delay, alpha, "TR")
    green = pd.read_parquet(green_path, columns=hcols + scols + trcols)
    frame = pd.concat([base, green], axis=1)
    baseline = base_groups(list(base.columns))["BP"]
    columns = {
        "M3_tail": baseline,
        "H_tail": baseline + hcols,
        "S_tail": baseline + hcols + scols,
        "TR_tail": baseline + hcols + scols + trcols,
    }
    valid_predictions = {}
    test_predictions = {}
    selection = {"delay": delay, "alpha": alpha, "models": {}}
    selection["window"] = args.window
    for name, selected in columns.items():
        valid_score, test_score, details, _ = fit_tail_selected(
            frame.iloc[train][selected], y[train],
            frame.iloc[valid][selected], y[valid],
            frame.iloc[test][selected], 0,
        )
        valid_predictions[name] = valid_score
        test_predictions[name] = test_score
        selection["models"][name] = details
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    best = None
    names = list(columns)
    valid_logits = {
        name: logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6))
        for name in names
    }
    test_logits = {
        name: logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6))
        for name in names
    }
    for candidate in product(weights, repeat=4):
        total = sum(candidate)
        if total == 0:
            continue
        score = expit(sum(w * valid_logits[n] for w, n in zip(candidate, names)) / total)
        objective = tail_objective(y[valid], score)
        if best is None or objective > best[0]:
            best = (objective, candidate, score)
    assert best is not None
    mixture_test = expit(
        sum(w * test_logits[n] for w, n in zip(best[1], names)) / sum(best[1])
    )
    valid_predictions["soft_mixture"] = best[2]
    test_predictions["soft_mixture"] = mixture_test
    selection["soft_mixture_weights"] = dict(zip(names, best[1]))
    summary_cols = [
        c for c in scols + trcols
        if c.startswith("agg__") or c.endswith(("__S_max", "__S_absdiff", "__TR_max_abs_update"))
    ]
    raw_valid = frame.iloc[valid][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_test = frame.iloc[test][summary_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    summary_valid = imputer.fit_transform(raw_valid)
    summary_test = imputer.transform(raw_test)
    meta_valid = np.column_stack([
        *[valid_logits[name] for name in names],
        summary_valid,
    ])
    meta_test = np.column_stack([
        *[test_logits[name] for name in names],
        summary_test,
    ])
    split = len(meta_valid) // 2
    best_stage = None
    for fraction in (0.05, 0.10, 0.20, 0.30):
        cutoff = np.quantile(valid_predictions["M3_tail"][:split], 1 - fraction)
        mask = valid_predictions["M3_tail"][:split] >= cutoff
        if np.unique(y[valid][:split][mask]).size < 2:
            continue
        model = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=2000, random_state=0
        )
        model.fit(meta_valid[:split][mask], y[valid][:split][mask])
        choose_base = valid_predictions["M3_tail"][split:]
        choose_mask = choose_base >= np.quantile(choose_base, 1 - fraction)
        choose_prob = model.predict_proba(meta_valid[split:])[:, 1]
        choose_score = rerank(choose_base, choose_prob, choose_mask)
        objective = tail_objective(y[valid][split:], choose_score)
        if best_stage is None or objective > best_stage[0]:
            best_stage = (objective, fraction)
    assert best_stage is not None
    fraction = best_stage[1]
    full_mask = valid_predictions["M3_tail"] >= np.quantile(
        valid_predictions["M3_tail"], 1 - fraction
    )
    final_model = LogisticRegression(
        class_weight="balanced", C=0.5, max_iter=2000, random_state=0
    )
    final_model.fit(meta_valid[full_mask], y[valid][full_mask])
    valid_prob = final_model.predict_proba(meta_valid)[:, 1]
    test_prob = final_model.predict_proba(meta_test)[:, 1]
    valid_predictions["two_stage"] = rerank(
        valid_predictions["M3_tail"], valid_prob, full_mask
    )
    test_mask = test_predictions["M3_tail"] >= np.quantile(
        test_predictions["M3_tail"], 1 - fraction
    )
    test_predictions["two_stage"] = rerank(
        test_predictions["M3_tail"], test_prob, test_mask
    )
    selection["two_stage_fraction"] = fraction
    metrics = {
        name: {
            "validation": evaluate(y[valid], valid_predictions[name]),
            "test": evaluate(y[test], test_predictions[name]),
        }
        for name in test_predictions
    }
    test_base = base.iloc[test].reset_index(drop=True)
    cohorts = cohort_metrics(test_base, y[test], test_predictions)
    bootstrap = paired_bootstrap(
        y[test],
        test_predictions["M3_tail"],
        {
            "H_tail": test_predictions["H_tail"],
            "S_tail": test_predictions["S_tail"],
            "TR_tail": test_predictions["TR_tail"],
            "soft_mixture": test_predictions["soft_mixture"],
            "two_stage": test_predictions["two_stage"],
        },
    )
    predictions = pd.DataFrame({
        "TransactionID": data.iloc[test]["TransactionID"].to_numpy(),
        "isFraud": y[test],
        "window": args.window,
        **{f"score_{name}": score for name, score in test_predictions.items()},
    })
    predictions.to_parquet(out / "tail_predictions_test.parquet", index=False)
    save_json(out / "tail_metrics.json", metrics)
    save_json(out / "metrics_optimized_exact.json", metrics)
    save_json(out / "tail_cohort_metrics.json", {"rows": cohorts})
    save_json(out / "tail_bootstrap.json", bootstrap)
    save_json(out / "tail_selected_hyperparams.json", selection)
    runtime = {
        "seconds": time.time() - started,
        "window": args.window,
        "optimized_exact_radius": 2,
        "optimized_exact_cap": 100,
        "fallback_rate": 0.0,
        "feature_runtime": feature_runtime,
    }
    save_json(out / "tail_runtime.json", runtime)
    save_json(out / "runtime_optimized_exact.json", runtime)
    print(json.dumps({name: value["test"] for name, value in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()

