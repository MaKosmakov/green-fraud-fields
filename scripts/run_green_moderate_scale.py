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

from green_fraud_fields.green_risk_field import GreenRiskFieldBuilder
from green_fraud_fields.ieee_cis import chronological_split, load_ieee_cis, tabular_features
from green_fraud_fields.modeling import evaluate, save_json
from green_fraud_fields.temporal_features import CausalFeatureBuilder
from run_green_risk_field import base_groups, cohort_metrics, green_columns, paired_bootstrap
from run_green_risk_tail import rerank
from run_ieee_tail_specialized import fit_tail_selected, tail_objective


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parquet_has_block_causal_metadata(path: Path) -> bool:
    if not path.exists():
        return False
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return (
        metadata.get(b"causal_policy") == b"strict_timestamp_block"
        and metadata.get(b"same_timestamp_policy") == b"block_frozen_t_minus"
        and metadata.get(b"label_release_policy") == b"release_time_strictly_less_than_candidate_timestamp"
    )


def write_base_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa

    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({
        b"causal_policy": b"strict_timestamp_block",
        b"same_timestamp_policy": b"block_frozen_t_minus",
        b"label_release_policy": b"release_time_strictly_less_than_candidate_timestamp",
        b"ego_radius": b"2",
        b"max_ego_nodes": b"100",
        b"cache_version": b"block_causal_v1",
    })
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def load_or_make_base(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists() and parquet_has_block_causal_metadata(path):
        return pd.read_parquet(path).reset_index(drop=True)
    if path.exists():
        path.unlink()
    base = pd.concat([
        tabular_features(data),
        CausalFeatureBuilder(
            compute_laplacian=False, ego_radius=2, max_ego_nodes=100
        ).transform(data),
    ], axis=1)
    base = base.loc[:, ~base.columns.duplicated()]
    write_base_parquet(base, path)
    return base


def load_or_make_green(
    data: pd.DataFrame,
    path: Path,
    delays: tuple[int, ...],
    alpha: int,
    force: bool,
) -> dict:
    if force and path.exists():
        path.unlink()
    if path.exists() and parquet_has_block_causal_metadata(path):
        return {"feature_seconds": None, "cached": True}
    if path.exists():
        path.unlink()
    builder = GreenRiskFieldBuilder(
        delays_days=delays,
        alphas=(alpha,),
        lambda_=0.1,
        ego_radius=2,
        max_ego_nodes=100,
        dense_threshold=150,
    )
    started = time.time()
    builder.write_parquet(data, str(path), batch_size=500)
    return {
        "feature_seconds": time.time() - started,
        "cached": False,
        **builder.runtime_summary(),
    }


def fit_model_set(
    frame: pd.DataFrame,
    y: np.ndarray,
    train: slice,
    valid: slice,
    test: slice,
    columns: dict[str, list[str]],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    valid_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    selection: dict = {}
    for name, selected in columns.items():
        valid_score, test_score, details, _ = fit_tail_selected(
            frame.iloc[train][selected], y[train],
            frame.iloc[valid][selected], y[valid],
            frame.iloc[test][selected], seed,
        )
        valid_predictions[name] = valid_score
        test_predictions[name] = test_score
        selection[name] = details
    return valid_predictions, test_predictions, selection


def add_soft_mixture(
    y_valid: np.ndarray,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    names: list[str],
) -> dict:
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    valid_logits = {
        name: logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6))
        for name in names
    }
    test_logits = {
        name: logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6))
        for name in names
    }
    best = None
    for candidate in product(weights, repeat=len(names)):
        total = sum(candidate)
        if total == 0:
            continue
        score = expit(sum(w * valid_logits[n] for w, n in zip(candidate, names)) / total)
        objective = tail_objective(y_valid, score)
        if best is None or objective > best[0]:
            best = (objective, candidate, score)
    assert best is not None
    test_score = expit(
        sum(w * test_logits[n] for w, n in zip(best[1], names)) / sum(best[1])
    )
    valid_predictions["soft_mixture"] = best[2]
    test_predictions["soft_mixture"] = test_score
    return {
        "components": names,
        "weights": dict(zip(names, best[1])),
        "validation": evaluate(y_valid, best[2]),
    }


def add_two_stage(
    frame: pd.DataFrame,
    y: np.ndarray,
    valid: slice,
    test: slice,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
) -> dict:
    raw_valid = frame.iloc[valid][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_test = frame.iloc[test][summary_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    summary_valid = imputer.fit_transform(raw_valid)
    summary_test = imputer.transform(raw_test)
    valid_logits = {
        name: logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6))
        for name in model_names
    }
    test_logits = {
        name: logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6))
        for name in model_names
    }
    meta_valid = np.column_stack([
        *[valid_logits[name] for name in model_names],
        summary_valid,
    ])
    meta_test = np.column_stack([
        *[test_logits[name] for name in model_names],
        summary_test,
    ])
    y_valid = y[valid]
    split = len(meta_valid) // 2
    best_stage = None
    for fraction in (0.05, 0.10, 0.20, 0.30):
        cutoff = np.quantile(valid_predictions["M3"][:split], 1 - fraction)
        mask = valid_predictions["M3"][:split] >= cutoff
        if np.unique(y_valid[:split][mask]).size < 2:
            continue
        model = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=2000, random_state=0
        )
        model.fit(meta_valid[:split][mask], y_valid[:split][mask])
        choose_base = valid_predictions["M3"][split:]
        choose_mask = choose_base >= np.quantile(choose_base, 1 - fraction)
        choose_prob = model.predict_proba(meta_valid[split:])[:, 1]
        choose_score = rerank(choose_base, choose_prob, choose_mask)
        objective = tail_objective(y_valid[split:], choose_score)
        if best_stage is None or objective > best_stage[0]:
            best_stage = (objective, fraction)
    if best_stage is None:
        valid_predictions["two_stage"] = valid_predictions["M3"].copy()
        test_predictions["two_stage"] = test_predictions["M3"].copy()
        return {"fallback": "M3", "reason": "insufficient positives in candidate sets"}
    fraction = best_stage[1]
    full_mask = valid_predictions["M3"] >= np.quantile(
        valid_predictions["M3"], 1 - fraction
    )
    final_model = LogisticRegression(
        class_weight="balanced", C=0.5, max_iter=2000, random_state=0
    )
    final_model.fit(meta_valid[full_mask], y_valid[full_mask])
    valid_prob = final_model.predict_proba(meta_valid)[:, 1]
    test_prob = final_model.predict_proba(meta_test)[:, 1]
    valid_predictions["two_stage"] = rerank(
        valid_predictions["M3"], valid_prob, full_mask
    )
    test_mask = test_predictions["M3"] >= np.quantile(
        test_predictions["M3"], 1 - fraction
    )
    test_predictions["two_stage"] = rerank(
        test_predictions["M3"], test_prob, test_mask
    )
    return {
        "candidate_fraction": fraction,
        "selection_objective": list(best_stage[0]),
        "second_half_validation": evaluate(
            y_valid[split:], valid_predictions["two_stage"][split:]
        ),
        "validation": evaluate(y_valid, valid_predictions["two_stage"]),
    }


def run_window(args, window_id: int, offset: int) -> dict:
    out = Path(args.out_dir) / f"window_{window_id}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = out / "status.json"

    def status(phase: str, **extra) -> None:
        save_json(status_path, {
            "window": window_id,
            "offset": offset,
            "phase": phase,
            "elapsed_seconds": time.time() - started,
            **extra,
        })

    status("loading_data")
    full = load_ieee_cis(args.data_dir, offset + args.window_size)
    data = full.iloc[offset:offset + args.window_size].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    status("building_base_features", rows=len(data))
    base = load_or_make_base(data, out / "base_features.parquet")
    delays = parse_ints(args.delays)
    alpha = args.alpha
    status("building_green_features", rows=len(data), delays=delays, alpha=alpha)
    feature_runtime = load_or_make_green(
        data, out / "green_features.parquet", delays, alpha, args.force_features
    )
    status("loading_green_features")
    schema = pq.ParquetFile(out / "green_features.parquet").schema.names
    h_by_delay = {delay: green_columns(schema, delay, alpha, "H") for delay in delays}
    s_by_delay = {delay: green_columns(schema, delay, alpha, "S") for delay in delays}
    hcols = [column for delay in delays for column in h_by_delay[delay]]
    scols = [column for delay in delays for column in s_by_delay[delay]]
    green = pd.read_parquet(out / "green_features.parquet", columns=hcols + scols)
    frame = pd.concat([base, green], axis=1)
    baseline = base_groups(list(base.columns))["BP"]
    primary_columns = {
        "M3": baseline,
        "H": baseline + hcols,
        "H_S": baseline + hcols + scols,
    }
    status("training_primary_models")
    valid_predictions, test_predictions, selection = fit_model_set(
        frame, y, train, valid, test, primary_columns, args.seed
    )
    status("training_soft_mixture")
    selection["soft_mixture"] = add_soft_mixture(
        y[valid], valid_predictions, test_predictions, ["M3", "H", "H_S"]
    )
    summary_cols = [
        c for c in scols
        if c.startswith("agg__") or c.endswith(("__S_max", "__S_absdiff"))
    ]
    status("training_two_stage")
    selection["two_stage"] = add_two_stage(
        frame, y, valid, test, valid_predictions, test_predictions,
        ["M3", "H", "H_S", "soft_mixture"], summary_cols,
    )
    per_delay_predictions = {}
    per_delay_selection = {}
    for delay in delays:
        status("training_per_delay_models", delay=delay)
        columns = {
            f"H_d{delay}": baseline + h_by_delay[delay],
            f"H_S_d{delay}": baseline + h_by_delay[delay] + s_by_delay[delay],
        }
        valid_d, test_d, selected_d = fit_model_set(
            frame, y, train, valid, test, columns, args.seed
        )
        valid_predictions.update(valid_d)
        test_predictions.update(test_d)
        per_delay_predictions.update(test_d)
        per_delay_selection.update(selected_d)
    selection["per_delay"] = per_delay_selection
    status("evaluating")
    metrics = {
        name: {
            "validation": evaluate(y[valid], valid_predictions[name]),
            "test": evaluate(y[test], test_predictions[name]),
        }
        for name in test_predictions
    }
    test_base = base.iloc[test].reset_index(drop=True)
    cohorts = cohort_metrics(test_base, y[test], {
        name: test_predictions[name]
        for name in ("M3", "H", "H_S", "soft_mixture", "two_stage")
    })
    bootstrap = paired_bootstrap(
        y[test],
        test_predictions["M3"],
        {
            "H": test_predictions["H"],
            "H_S": test_predictions["H_S"],
            "soft_mixture": test_predictions["soft_mixture"],
            "two_stage": test_predictions["two_stage"],
        },
        replicates=args.bootstrap_replicates,
        block_size=500,
    )
    predictions = pd.DataFrame({
        "TransactionID": data.iloc[test]["TransactionID"].to_numpy(),
        "TransactionDT": data.iloc[test]["TransactionDT"].to_numpy(),
        "isFraud": y[test],
        "window": window_id,
        **{
            f"score_{name}": test_predictions[name]
            for name in ("M3", "H", "H_S", "soft_mixture", "two_stage")
        },
    })
    predictions.to_parquet(out / "predictions_test.parquet", index=False)
    save_json(out / "metrics.json", metrics)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    save_json(out / "bootstrap.json", bootstrap)
    save_json(out / "selection.json", selection)
    runtime = {
        "seconds": time.time() - started,
        "window": window_id,
        "offset": offset,
        "rows": len(data),
        "split": {"train": len(range(*train.indices(len(data)))), "valid": len(range(*valid.indices(len(data)))), "test": len(range(*test.indices(len(data))))},
        "delays": delays,
        "alpha": alpha,
        "dense_exact_radius": 2,
        "dense_exact_cap": 100,
        "fallback_rate": 0.0,
        "feature_runtime": feature_runtime,
        "random_forest": "not used",
    }
    save_json(out / "runtime.json", runtime)
    summary = []
    for name in ("M3", "H", "H_S", "soft_mixture", "two_stage"):
        test_metrics = metrics[name]["test"]
        summary.append({
            "window": window_id,
            "model": name,
            "auc_pr": test_metrics["auc_pr"],
            "precision_at_0.005": test_metrics["precision_at_0.005"],
            "precision_at_0.01": test_metrics["precision_at_0.01"],
            "precision_at_0.02": test_metrics["precision_at_0.02"],
            "precision_at_0.05": test_metrics["precision_at_0.05"],
            "runtime_seconds": runtime["seconds"],
            "feature_seconds": feature_runtime.get("feature_seconds"),
        })
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    status("complete", runtime_seconds=runtime["seconds"])
    print(json.dumps({row["model"]: row for row in summary}, indent=2))
    return {"runtime": runtime, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--out-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--start-windows", default="0,1")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    runtimes = {}
    for window_id in parse_ints(args.start_windows):
        result = run_window(args, window_id, window_id * args.window_size)
        runtimes[f"window_{window_id}"] = result["runtime"]
        all_rows.extend(result["summary"])
    pd.DataFrame(all_rows).to_csv(root / "window_summary.csv", index=False)
    save_json(root / "runtime.json", runtimes)


if __name__ == "__main__":
    main()

