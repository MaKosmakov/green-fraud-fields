from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from green_fraud_fields.green_risk_field import GreenRiskFieldBuilder
from green_fraud_fields.ieee_cis import (
    build_transaction_edges,
    chronological_split,
    load_ieee_cis_cached,
    tabular_features,
    transaction_edge_records,
)
from green_fraud_fields.modeling import evaluate, save_json
from green_fraud_fields.temporal_features import CausalFeatureBuilder
from green_fraud_fields.green_risk_field import RISK_LAYERS
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached
from run_green_adaptive_theory import PrecisionWeightedGreenBuilder
from run_green_crossfit_reranker import oof_base_predictions
from run_green_focused_improvements import tuned_soft_mixture
from run_green_risk_field import base_groups, cohort_metrics, green_columns
from run_green_risk_tail import rerank


METRICS = [
    "auc_pr",
    "roc_auc",
    "precision_at_0.005",
    "precision_at_0.01",
    "precision_at_0.02",
    "precision_at_0.05",
]

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def strict_metadata(extra: dict[str, str | int | float]) -> dict[bytes, bytes]:
    metadata = {
        b"causal_policy": b"strict_timestamp_block",
        b"same_timestamp_policy": b"block_frozen_t_minus",
        b"label_release_policy": b"release_time_strictly_less_than_candidate_timestamp",
        b"warm_start_policy": b"prefix_history_from_all_prior_rows",
        b"cache_version": b"warmstart_block_causal",
    }
    metadata.update({str(k).encode(): str(v).encode() for k, v in extra.items()})
    return metadata


def has_warmstart_metadata(path: Path, *, feature_family: str, window: int) -> bool:
    if not path.exists():
        return False
    metadata = pq.ParquetFile(path).metadata.metadata or {}
    return (
        metadata.get(b"causal_policy") == b"strict_timestamp_block"
        and metadata.get(b"same_timestamp_policy") == b"block_frozen_t_minus"
        and metadata.get(b"label_release_policy") == b"release_time_strictly_less_than_candidate_timestamp"
        and metadata.get(b"warm_start_policy") == b"prefix_history_from_all_prior_rows"
        and metadata.get(b"feature_family") == feature_family.encode()
        and metadata.get(b"window") == str(window).encode()
    )


def write_feature_parquet(frame: pd.DataFrame, path: Path, metadata: dict[bytes, bytes]) -> None:
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    merged = dict(table.schema.metadata or {})
    merged.update(metadata)
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, path, compression="zstd")


def stream_selected_records(
    builder,
    records: list[dict],
    score_start: int,
    score_end: int,
) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    selected: list[dict] = []
    current_time = None
    block: list[tuple[int, dict]] = []

    def warm_commit(warm_block: list[tuple[int, dict]]) -> None:
        """Advance builder state for pre-window blocks without solving features."""
        if not warm_block:
            return
        rows = [row for _, row in warm_block]
        if isinstance(builder, CausalFeatureBuilder):
            for row in rows:
                amount = float(row.get("TransactionAmt", 0.0) or 0.0)
                for edge in build_transaction_edges(row):
                    builder._insert(edge, amount)
            return
        if isinstance(builder, GreenRiskFieldBuilder):
            rows_and_edges = [(row, build_transaction_edges(row)) for row in rows]
            builder._commit_block(rows_and_edges)
            return
        if isinstance(builder, PrecisionWeightedGreenBuilder):
            for row in rows:
                edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
                for edge in edges:
                    builder.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
                builder.release.schedule(float(row["TransactionDT"]), int(row["isFraud"]), edges)
            return
        raise TypeError(f"Unsupported warm-start builder: {type(builder)!r}")

    def flush() -> None:
        if not block:
            return
        if block[-1][0] < score_start:
            warm_commit(block)
            return
        outputs = builder.process_timestamp_block([row for _, row in block])
        for (idx, _), output in zip(block, outputs, strict=True):
            if score_start <= idx < score_end:
                selected.append(output)

    for idx, row in enumerate(records):
        now = float(row["TransactionDT"])
        if current_time is None:
            current_time = now
        if now != current_time:
            flush()
            block = []
            current_time = now
        block.append((idx, row))
    flush()
    return pd.DataFrame(selected).reset_index(drop=True), {
        "seconds": time.time() - started,
        "selected_rows": len(selected),
        "prefix_rows": len(records),
    }


def build_warmstart_features(args: argparse.Namespace, window: int) -> dict:
    started = time.time()
    root = Path(args.out_dir) / "00_warmstart_features" / f"window_{window}"
    root.mkdir(parents=True, exist_ok=True)
    score_start = window * args.window_size
    score_end = (window + 1) * args.window_size
    full = load_ieee_cis_cached(
        args.data_dir,
        score_end,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    ).reset_index(drop=True)
    scored = full.iloc[score_start:score_end].reset_index(drop=True)
    runtime: dict[str, object] = {
        "window": window,
        "score_start": score_start,
        "score_end": score_end,
        "prefix_rows": int(len(full)),
        "score_rows": int(len(scored)),
        "warm_start_policy": "prefix_history_from_all_prior_rows",
    }

    base_path = root / "base_features.parquet"
    if base_path.exists() and not has_warmstart_metadata(base_path, feature_family="base", window=window):
        base_path.unlink()
    if base_path.exists() and not args.force_features:
        runtime["base"] = {"cached": True}
    else:
        base_builder = CausalFeatureBuilder(compute_laplacian=False, ego_radius=args.radius, max_ego_nodes=args.cap)
        base_records = transaction_edge_records(full, include_amount=True)
        causal, causal_runtime = stream_selected_records(base_builder, base_records, score_start, score_end)
        base = pd.concat([tabular_features(scored), causal], axis=1)
        base = base.loc[:, ~base.columns.duplicated()]
        write_feature_parquet(
            base,
            base_path,
            strict_metadata(
                {
                    "feature_family": "base",
                    "window": window,
                    "score_start": score_start,
                    "score_end": score_end,
                    "ego_radius": args.radius,
                    "max_ego_nodes": args.cap,
                }
            ),
        )
        runtime["base"] = {"cached": False, **causal_runtime, "columns": int(base.shape[1])}

    green_path = root / "green_features.parquet"
    if green_path.exists() and not has_warmstart_metadata(green_path, feature_family="plain_green", window=window):
        green_path.unlink()
    if green_path.exists() and not args.force_features:
        runtime["green"] = {"cached": True}
    else:
        green_builder = GreenRiskFieldBuilder(
            delays_days=parse_ints(args.delays),
            alphas=(args.alpha,),
            lambda_=args.lambda_,
            ego_radius=args.radius,
            max_ego_nodes=args.cap,
            dense_threshold=150,
        )
        green_records = transaction_edge_records(full, include_label=True)
        green, green_runtime = stream_selected_records(green_builder, green_records, score_start, score_end)
        green = green.astype("float32")
        write_feature_parquet(
            green,
            green_path,
            strict_metadata(
                {
                    "feature_family": "plain_green",
                    "window": window,
                    "score_start": score_start,
                    "score_end": score_end,
                    "delays": args.delays,
                    "alpha": args.alpha,
                    "lambda": args.lambda_,
                    "ego_radius": args.radius,
                    "max_ego_nodes": args.cap,
                }
            ),
        )
        runtime["green"] = {
            "cached": False,
            **green_runtime,
            **green_builder.runtime_summary(),
            "columns": int(green.shape[1]),
        }

    precision_path = root / "precision_weighted_green.parquet"
    if precision_path.exists() and not has_warmstart_metadata(precision_path, feature_family="precision_green", window=window):
        precision_path.unlink()
    if precision_path.exists() and not args.force_features:
        runtime["precision"] = {"cached": True}
    else:
        precision_builder = PrecisionWeightedGreenBuilder(
            delay_days=args.delay,
            history_alpha=args.alpha,
            precision_alpha=args.precision_alpha,
            radius=args.radius,
            cap=args.cap,
        )
        precision_records = transaction_edge_records(full, include_label=True)
        precision, precision_runtime = stream_selected_records(precision_builder, precision_records, score_start, score_end)
        precision = precision.astype("float32")
        write_feature_parquet(
            precision,
            precision_path,
            strict_metadata(
                {
                    "feature_family": "precision_green",
                    "window": window,
                    "score_start": score_start,
                    "score_end": score_end,
                    "delay_days": args.delay,
                    "alpha": args.alpha,
                    "precision_alpha": args.precision_alpha,
                    "ego_radius": args.radius,
                    "max_ego_nodes": args.cap,
                }
            ),
        )
        runtime["precision"] = {
            "cached": False,
            **precision_runtime,
            "edge_solves": int(precision_builder.counts["edge_solves"]),
            "mean_ego_size": float(np.mean(precision_builder.ego_sizes)) if precision_builder.ego_sizes else 0.0,
            "max_ego_size": int(max(precision_builder.ego_sizes)) if precision_builder.ego_sizes else 0,
            "columns": int(precision.shape[1]),
        }

    runtime["seconds"] = time.time() - started
    save_json(root / "feature_runtime.json", runtime)
    return runtime


def safe_logit(score: np.ndarray) -> np.ndarray:
    return logit(np.clip(score, 1e-6, 1 - 1e-6))


def meta_matrices(
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
    return (
        np.column_stack([*[safe_logit(oof_predictions[name]) for name in model_names], summary_oof]),
        np.column_stack([*[safe_logit(valid_predictions[name]) for name in model_names], summary_valid]),
        np.column_stack([*[safe_logit(test_predictions[name]) for name in model_names], summary_test]),
    )


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
    *,
    threshold_transfer: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta_oof, meta_valid, meta_test = meta_matrices(
        frame, oof_index, valid, test, oof_predictions, valid_predictions, test_predictions, model_names, summary_cols
    )
    y_oof = y[oof_index]
    y_valid = y[valid]
    best = None
    trials = []
    for fraction in fractions:
        oof_cutoff = float(np.quantile(oof_predictions["M3"], 1 - fraction))
        train_mask = oof_predictions["M3"] >= oof_cutoff
        if np.unique(y_oof[train_mask]).size < 2:
            continue
        model = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0)
        model.fit(meta_oof[train_mask], y_oof[train_mask])
        valid_cutoff = float(np.quantile(valid_predictions["M3"], 1 - fraction))
        valid_mask = valid_predictions["M3"] >= valid_cutoff
        valid_probability = model.predict_proba(meta_valid)[:, 1]
        valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
        metrics = evaluate(y_valid, valid_score)
        key = (metrics["precision_at_0.01"], metrics["precision_at_0.005"], metrics["auc_pr"])
        trials.append(
            {
                "fraction": fraction,
                "oof_cutoff": oof_cutoff,
                "valid_cutoff": valid_cutoff,
                "selection_key": list(key),
                "validation": metrics,
            }
        )
        if best is None or key > best[0]:
            best = (key, fraction, model, valid_cutoff)
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in OOF tail candidates",
            "test_used_for_selection": False,
        }
    _, fraction, model, valid_cutoff = best
    valid_probability = model.predict_proba(meta_valid)[:, 1]
    test_probability = model.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= valid_cutoff
    if threshold_transfer:
        test_mask = test_predictions["M3"] >= valid_cutoff
        test_cutoff = valid_cutoff
        policy = "validation_absolute_m3_threshold_transferred_to_test"
    else:
        test_cutoff = float(np.quantile(test_predictions["M3"], 1 - fraction))
        test_mask = test_predictions["M3"] >= test_cutoff
        policy = "test_batch_top_fraction_label_free_quantile"
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "OOF-trained logistic reranker, validation-selected candidate fraction by P@1%, P@0.5%, AUC-PR",
        "candidate_fraction": fraction,
        "valid_cutoff": valid_cutoff,
        "test_cutoff": test_cutoff,
        "test_region_policy": policy,
        "selection_key": list(best[0]),
        "trials": trials,
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


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
    build_runtime = build_warmstart_features(args, window)
    timer.mark("build_or_load_features")
    out = Path(args.out_dir) / "01_warmstart_graph_vs_history" / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    feature_window = Path(args.out_dir) / "00_warmstart_features" / f"window_{window}"
    score_end = (window + 1) * args.window_size
    score_start = window * args.window_size
    full = load_ieee_cis_cached(
        args.data_dir,
        score_end,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    )
    data = full.iloc[score_start:score_end].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    base = pd.read_parquet(feature_window / "base_features.parquet").reset_index(drop=True)
    green_path = feature_window / "green_features.parquet"
    schema = pq.ParquetFile(green_path).schema.names
    hcols = green_columns(schema, args.delay, args.alpha, "H")
    scols = green_columns(schema, args.delay, args.alpha, "S")
    green = pd.read_parquet(green_path, columns=hcols + scols).reset_index(drop=True)
    sd_path = feature_window / "precision_weighted_green.parquet"
    sd_schema = pq.ParquetFile(sd_path).schema.names
    sd_cols_all = [c for c in sd_schema if "__SD_" in c or "__SD" in c]
    sd_sqrt_cols = [c for c in sd_cols_all if "SD_sqrt" in c]
    sd = pd.read_parquet(sd_path, columns=sd_sqrt_cols).reset_index(drop=True)
    frame = pd.concat([base, green, sd], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    timer.mark("read_and_assemble")

    baseline_cols = base_groups(list(base.columns))["BP"]
    model_cols = {
        "M3": baseline_cols,
        "M3_H_raw": baseline_cols + hcols,
        "M3_H_raw_S_D": baseline_cols + hcols + sd_sqrt_cols,
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

    mix_valid, mix_test, mix_selection = tuned_soft_mixture(
        y[valid], valid_predictions, test_predictions, ["M3", "M3_H_raw", "M3_H_raw_S_D"]
    )
    valid_predictions["adaptive_soft"] = mix_valid
    test_predictions["adaptive_soft"] = mix_test
    selection["adaptive_soft"] = mix_selection

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
    selection["_oof"] = oof_details
    summary_cols = [c for c in sd_sqrt_cols if c.startswith("agg__") or "all_" in c]
    fractions = parse_floats(args.tail_fractions)
    for name, transfer in [
        ("crossfit_logistic_tail", False),
        ("crossfit_logistic_tail_threshold_transfer", True),
    ]:
        vscore, tscore, details = crossfit_logistic_tail(
            frame,
            y,
            oof_index,
            valid,
            test,
            oof_predictions,
            valid_predictions,
            test_predictions,
            ["M3", "M3_H_raw", "M3_H_raw_S_D"],
            summary_cols,
            fractions,
            threshold_transfer=transfer,
        )
        valid_predictions[name] = vscore
        test_predictions[name] = tscore
        selection[name] = details
    timer.mark("soft_and_crossfit_tail")

    metrics = {
        name: {"validation": evaluate(y[valid], valid_predictions[name]), "test": evaluate(y[test], test_predictions[name])}
        for name in valid_predictions
    }
    base_test = base.iloc[test].reset_index(drop=True)
    cohorts = cohort_metrics(base_test, y[test], test_predictions)
    runtime = {
        "seconds": timer.total(),
        "window": window,
        "stage_seconds": timer.stages,
        "feature_runtime": build_runtime,
        "model_fit_timing": model_timing,
        "warm_start_policy": "prefix_history_from_all_prior_rows",
        "causal_policy": "strict_timestamp_block",
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
    }
    rows = summary_rows(window, metrics, runtime["seconds"])
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    save_json(out / "runtime.json", runtime)
    expected_ids = data.iloc[test]["TransactionID"].to_numpy()
    pd.DataFrame({"TransactionID": expected_ids, **{f"score_{k}": v for k, v in test_predictions.items()}}).to_parquet(
        out / "predictions_test.parquet", index=False
    )
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
        if len(common) == 0:
            continue
        row = {"model": model, "baseline": baseline_model, "windows": int(len(common))}
        for metric in METRICS:
            delta = group.loc[common, metric] - baseline.loc[common, metric]
            row[f"mean_gain_{metric}"] = float(delta.mean())
            row[f"win_count_{metric}"] = int((delta > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_gain_precision_at_0.01", "mean_gain_auc_pr"], ascending=False)


def aggregate_vs_v2(summary: pd.DataFrame, v2_path: Path) -> pd.DataFrame:
    if not v2_path.exists():
        return pd.DataFrame()
    v2 = pd.read_csv(v2_path)
    rows = []
    for model, group in summary.groupby("model"):
        group = group.set_index("window").sort_index()
        for v2_model in ["M3_H_raw_S_D", "adaptive_two_stage", "crossfit_logistic_tail"]:
            base = v2[v2["model"] == v2_model].set_index("window")
            common = group.index.intersection(base.index)
            if len(common) == 0:
                continue
            row = {"model": model, "v2_baseline": v2_model, "windows": int(len(common))}
            for metric in METRICS:
                delta = group.loc[common, metric] - base.loc[common, metric]
                row[f"mean_delta_{metric}"] = float(delta.mean())
                row[f"win_count_{metric}"] = int((delta > 0).sum())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["v2_baseline", "mean_delta_precision_at_0.01"], ascending=[True, False])


def write_aggregate_outputs(args: argparse.Namespace) -> None:
    root = Path(args.out_dir) / "01_warmstart_graph_vs_history"
    summaries = []
    cohorts = []
    runtimes = {}
    for path in sorted(root.glob("window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(root.glob("window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            row["window"] = window
            cohorts.append(row)
    for path in sorted(root.glob("window_*/runtime.json")):
        runtimes[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    summary.to_csv(root / "window_summary.csv", index=False)
    cohort_frame.to_csv(root / "cohort_metrics.csv", index=False)
    save_json(root / "runtime.json", runtimes)
    aggregate_gains(summary, "M3").to_csv(root / "mean_gains_vs_m3.csv", index=False)
    aggregate_gains(summary, "M3_H_raw").to_csv(root / "mean_gains_vs_history.csv", index=False)
    aggregate_vs_v2(summary, Path(args.v2_summary)).to_csv(root / "mean_deltas_vs_v2.csv", index=False)
    report = [
        "# Warm-start block-causal run",
        "",
        "Strict timestamp-block features with graph/history warm-started from all prior rows in the global chronology.",
        "No Random Forest and no test-label tuning.",
        "",
        "## Mean gains over M3",
        "",
        aggregate_gains(summary, "M3").to_csv(index=False),
        "",
        "## Mean gains over raw history",
        "",
        aggregate_gains(summary, "M3_H_raw").to_csv(index=False),
        "",
        "## Mean deltas versus v2",
        "",
        aggregate_vs_v2(summary, Path(args.v2_summary)).to_csv(index=False),
    ]
    (root / "warmstart_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs/ieee_green_warmstart_block_causal")
    parser.add_argument("--v2-summary", default="outputs/ieee_green_final_review_gates_v2_block_causal/01_graph_vs_history/window_summary.csv")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--precision-alpha", type=float, default=5.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tail-fractions", default="0.025,0.05,0.10,0.20,0.30")
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--oof-warmup-fraction", type=float, default=0.2)
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--force-oof", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        write_aggregate_outputs(args)
        return
    for window in parse_ints(args.windows):
        run_window(args, window)
    write_aggregate_outputs(args)


if __name__ == "__main__":
    main()
