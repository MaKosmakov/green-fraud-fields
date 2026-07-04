from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from math import log1p
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.green_risk_field import DelayedReleaseState, RISK_LAYERS
from green_fraud_fields.ieee_cis import (
    build_transaction_edges,
    chronological_split,
    load_ieee_cis_cached,
    transaction_edge_records,
)
from green_fraud_fields.laplacian_features import LayerGraph
from green_fraud_fields.modeling import evaluate, save_json
from run_green_adaptive_next_round import StageTimer, fit_named_models_cached
from run_green_adaptive_theory import dense_laplacian
from run_green_review_graph_history import load_reference_predictions
from run_green_risk_field import base_groups, cohort_metrics, green_columns


METRICS = [
    "auc_pr",
    "roc_auc",
    "precision_at_0.005",
    "precision_at_0.01",
    "precision_at_0.02",
    "precision_at_0.05",
]

RUN_SUBDIR = "05_posterior_uncertainty"

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


class PosteriorUncertaintyGreenBuilder:
    """Strict timestamp-block posterior mean/variance features for sqrt-precision S_D.

    This is intentionally a frozen diagnostic, not a feature-search grid.  It uses
    the same causal update order and radius-2/cap-100 dense exact ego solve as the
    v2 block-causal headline, but augments the posterior mean with endpoint
    posterior variance and a simple z-score.
    """

    def __init__(
        self,
        delay_days: int = 0,
        history_alpha: int = 5,
        precision_alpha: float = 5.0,
        radius: int = 2,
        cap: int = 100,
        eps: float = 1e-8,
    ) -> None:
        self.delay_days = delay_days
        self.history_alpha = history_alpha
        self.precision_alpha = precision_alpha
        self.radius = radius
        self.cap = cap
        self.eps = eps
        self.release = DelayedReleaseState(delay_days * 86400.0)
        self.graphs: dict[str, LayerGraph] = defaultdict(LayerGraph)
        self.counts = defaultdict(int)
        self.ego_sizes: list[int] = []

    def _endpoint_features(self, prefix: str, left: float, right: float) -> dict[str, float]:
        return {
            f"{prefix}_left": left,
            f"{prefix}_right": right,
            f"{prefix}_sum": left + right,
            f"{prefix}_max": max(left, right),
            f"{prefix}_min": min(left, right),
            f"{prefix}_diff": left - right,
            f"{prefix}_absdiff": abs(left - right),
        }

    def _solve_edge(self, edge) -> dict[str, float]:
        graph = self.graphs[edge.layer]
        history = self.release.history_by_layer[edge.layer]
        nodes = graph.ego_nodes((edge.a, edge.b), radius=self.radius, max_nodes=self.cap)
        laplacian, index = dense_laplacian(graph, nodes)
        h = np.array([history.signal(node, self.history_alpha) for node in nodes], dtype=np.float64)
        exposure = np.array([history.exposure[node] for node in nodes], dtype=np.float64)
        d = self.precision_alpha + np.sqrt(exposure)
        system = laplacian.copy()
        system.flat[:: len(nodes) + 1] += d

        left_index = index[edge.a]
        right_index = index[edge.b]
        rhs = np.zeros((len(nodes), 3), dtype=np.float64)
        rhs[:, 0] = d * h
        rhs[left_index, 1] = 1.0
        rhs[right_index, 2] = 1.0

        factor = cho_factor(system, lower=False, check_finite=False)
        solved = cho_solve(factor, rhs, check_finite=False)
        sd_left = float(solved[left_index, 0])
        sd_right = float(solved[right_index, 0])
        var_left = max(float(solved[left_index, 1]), 0.0)
        var_right = max(float(solved[right_index, 2]), 0.0)
        z_left = sd_left / np.sqrt(var_left + self.eps)
        z_right = sd_right / np.sqrt(var_right + self.eps)

        outputs: dict[str, float] = {
            "H_left": history.signal(edge.a, self.history_alpha),
            "H_right": history.signal(edge.b, self.history_alpha),
            "H_left_exposure_log": log1p(history.exposure[edge.a]),
            "H_right_exposure_log": log1p(history.exposure[edge.b]),
        }
        outputs.update(self._endpoint_features("SD_sqrt", sd_left, sd_right))
        outputs.update(self._endpoint_features("VAR_sqrt", var_left, var_right))
        outputs.update(self._endpoint_features("Z_sqrt", z_left, z_right))
        self.ego_sizes.append(len(nodes))
        self.counts["edge_solves"] += 1
        return outputs

    def _score_row_from_frozen_state(self, row: dict) -> tuple[dict[str, float], list]:
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        per_layer = {edge.layer: self._solve_edge(edge) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({f"{safe}__d{self.delay_days}_a{self.history_alpha}__{k}": v for k, v in values.items()})
        for rule in ("SD_sqrt", "VAR_sqrt", "Z_sqrt"):
            left = [v[f"{rule}_left"] for v in per_layer.values()]
            right = [v[f"{rule}_right"] for v in per_layer.values()]
            absdiff = [v[f"{rule}_absdiff"] for v in per_layer.values()]
            if not left:
                continue
            endpoints = left + right
            output.update(
                {
                    f"agg__d{self.delay_days}_a{self.history_alpha}__{rule}_all_sum": float(np.sum(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__{rule}_all_max": float(np.max(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__{rule}_all_mean": float(np.mean(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__{rule}_all_absdiff_max": float(np.max(absdiff)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__{rule}_all_absdiff_mean": float(np.mean(absdiff)),
                }
            )
        return output, edges

    def process_timestamp_block(self, block: list[dict]) -> list[dict[str, float]]:
        if not block:
            return []
        now = float(block[0]["TransactionDT"])
        self.release.release_before(now)
        scored: list[dict[str, float]] = []
        pending: list[tuple[dict, list]] = []
        for row in block:
            output, edges = self._score_row_from_frozen_state(row)
            scored.append(output)
            pending.append((row, edges))
        for row, edges in pending:
            for edge in edges:
                self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
            self.release.schedule(float(row["TransactionDT"]), int(row["isFraud"]), edges)
        return scored

    def write_parquet(self, df: pd.DataFrame, path: Path, batch_size: int = 500) -> dict:
        started = time.time()
        writer = None
        columns = None
        pending_records: list[dict[str, float]] = []

        def write_records(records: list[dict[str, float]]) -> None:
            nonlocal writer, columns
            if not records:
                return
            batch = pd.DataFrame(records).astype("float32")
            if columns is None:
                columns = list(batch.columns)
            else:
                batch = batch.reindex(columns=columns)
            table = pa.Table.from_pandas(batch, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata.update(
                {
                    b"causal_policy": b"strict_timestamp_block",
                    b"same_timestamp_policy": b"block_frozen_t_minus",
                    b"label_release_policy": b"release_time_strictly_less_than_candidate_timestamp",
                    b"feature_family": b"posterior_uncertainty_sqrt_precision",
                    b"delay_days": str(self.delay_days).encode(),
                    b"history_alpha": str(self.history_alpha).encode(),
                    b"precision_alpha": str(self.precision_alpha).encode(),
                    b"ego_radius": str(self.radius).encode(),
                    b"max_ego_nodes": str(self.cap).encode(),
                    b"cache_version": b"block_causal_posterior_v1",
                }
            )
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)

        try:
            current_time = None
            block: list[dict] = []
            for row in transaction_edge_records(df, include_label=True):
                now = float(row["TransactionDT"])
                if current_time is None:
                    current_time = now
                if now != current_time:
                    pending_records.extend(self.process_timestamp_block(block))
                    if len(pending_records) >= batch_size:
                        write_records(pending_records)
                        pending_records = []
                    block = []
                    current_time = now
                block.append(row)
            pending_records.extend(self.process_timestamp_block(block))
            write_records(pending_records)
        finally:
            if writer is not None:
                writer.close()
        return {
            "feature_seconds": time.time() - started,
            "edge_solves": int(self.counts["edge_solves"]),
            "mean_ego_size": float(np.mean(self.ego_sizes)) if self.ego_sizes else 0.0,
            "max_ego_size": int(max(self.ego_sizes)) if self.ego_sizes else 0,
            "causal_policy": "strict_timestamp_block",
            "feature_family": "posterior_uncertainty_sqrt_precision",
        }


def require_block_causal(path: Path) -> None:
    metadata = pq.ParquetFile(path).metadata.metadata or {}
    if metadata.get(b"causal_policy") != b"strict_timestamp_block":
        raise ValueError(f"{path} is not tagged causal_policy=strict_timestamp_block")


def posterior_columns(columns: list[str], prefix: str) -> list[str]:
    return [c for c in columns if f"__{prefix}_" in c or c.startswith(f"agg__") and f"__{prefix}_" in c]


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
    root = Path(args.out_dir) / RUN_SUBDIR
    out = root / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)

    baseline_window = Path(args.baseline_dir) / f"window_{window}"
    crossfit_window = Path(args.crossfit_dir) / f"window_{window}"
    base_path = baseline_window / "base_features.parquet"
    green_path = baseline_window / "green_features.parquet"
    require_block_causal(base_path)
    require_block_causal(green_path)

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

    base = pd.read_parquet(base_path).reset_index(drop=True)
    green_schema = pq.ParquetFile(green_path).schema.names
    hcols = green_columns(green_schema, args.delay, args.alpha, "H")
    green = pd.read_parquet(green_path, columns=hcols).reset_index(drop=True)
    timer.mark("read_base_history")

    posterior_path = out / "posterior_uncertainty_green.parquet"
    if posterior_path.exists() and not args.force_features:
        require_block_causal(posterior_path)
        posterior_runtime = {"cached": True}
    else:
        builder = PosteriorUncertaintyGreenBuilder(
            delay_days=args.delay,
            history_alpha=args.alpha,
            precision_alpha=args.precision_alpha,
            radius=args.radius,
            cap=args.cap,
        )
        posterior_runtime = builder.write_parquet(data, posterior_path)
        posterior_runtime["cached"] = False
    posterior = pd.read_parquet(posterior_path).reset_index(drop=True)
    posterior_schema = list(posterior.columns)
    sd_cols = posterior_columns(posterior_schema, "SD_sqrt")
    var_cols = posterior_columns(posterior_schema, "VAR_sqrt")
    z_cols = posterior_columns(posterior_schema, "Z_sqrt")
    timer.mark("posterior_features")

    frame = pd.concat([base, green, posterior[sd_cols + var_cols + z_cols]], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    baseline_cols = base_groups(list(base.columns))["BP"]
    model_cols = {
        "M3": baseline_cols,
        "M3_H_raw": baseline_cols + hcols,
        "M3_H_raw_S_D": baseline_cols + hcols + sd_cols,
        "M3_H_raw_S_D_sigma2": baseline_cols + hcols + sd_cols + var_cols,
        "M3_H_raw_S_D_z": baseline_cols + hcols + sd_cols + z_cols,
        "M3_H_raw_S_D_sigma2_z": baseline_cols + hcols + sd_cols + var_cols + z_cols,
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
    timer.mark("fit_uncertainty_panel")

    reference = load_reference_predictions(crossfit_window / "predictions_test.parquet")
    expected_ids = data.iloc[test]["TransactionID"].to_numpy()
    if not np.array_equal(reference["TransactionID"].to_numpy(), expected_ids):
        raise ValueError(f"Reference predictions for window {window} do not align with the chronological test slice")
    test_predictions["adaptive_two_stage"] = reference["score_split_valid_logistic_tail"].to_numpy(float)
    selection["adaptive_two_stage"] = {
        "source": str(crossfit_window / "predictions_test.parquet"),
        "test_used_for_selection": False,
        "reference": "current_v2_headline",
    }
    timer.mark("load_headline_reference")

    metrics = {}
    for model, score in test_predictions.items():
        if model in valid_predictions:
            validation = evaluate(y[valid], valid_predictions[model])
        else:
            validation = {"reference_score_only": True}
        metrics[model] = {"validation": validation, "test": evaluate(y[test], score)}

    base_test = base.iloc[test].reset_index(drop=True)
    y_test = y[test]
    cohorts = cohort_metrics(base_test, y_test, test_predictions)
    timer.mark("evaluate")

    runtime = {
        "seconds": timer.total(),
        "window": window,
        "stage_seconds": timer.stages,
        "posterior_runtime": posterior_runtime,
        "uses_cached_dense_exact_green": True,
        "causal_policy": "strict_timestamp_block",
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
        "frozen_panel": ["S_D", "S_D+sigma2", "S_D+z", "S_D+sigma2+z"],
    }
    rows = summary_rows(window, metrics, runtime["seconds"])
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    save_json(out / "runtime.json", runtime)
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
        out = {"model": model, "baseline": baseline_model, "windows": int(len(common))}
        for metric in METRICS:
            delta = group.loc[common, metric] - baseline.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(delta.mean())
            out[f"win_count_{metric}"] = int((delta > 0).sum())
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mean_gain_precision_at_0.01", "mean_gain_auc_pr"], ascending=False)


def critical_cohort_gains(cohorts: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    rows = []
    critical = cohorts[cohorts["cohort"].isin(["known_endpoints", "C00_newedge"])].copy()
    for (cohort, model), group in critical.groupby(["cohort", "model"]):
        if model == baseline_model:
            continue
        base = critical[(critical["cohort"] == cohort) & (critical["model"] == baseline_model)].set_index("window")
        group = group.set_index("window").sort_index()
        common = group.index.intersection(base.index)
        if len(common) == 0:
            continue
        out = {"cohort": cohort, "model": model, "baseline": baseline_model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            delta = group.loc[common, metric] - base.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(delta.mean())
            out[f"win_count_{metric}"] = int((delta > 0).sum())
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["cohort", "mean_gain_precision_at_0.01"], ascending=[True, False])


def write_aggregate_outputs(root: Path) -> None:
    run_root = root / RUN_SUBDIR
    summaries = []
    cohorts = []
    runtimes = {}
    for path in sorted(run_root.glob("window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(run_root.glob("window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            row["window"] = window
            cohorts.append(row)
    for path in sorted(run_root.glob("window_*/runtime.json")):
        runtimes[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))

    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    summary.to_csv(run_root / "window_summary.csv", index=False)
    cohort_frame.to_csv(run_root / "cohort_metrics.csv", index=False)
    save_json(run_root / "runtime.json", runtimes)

    gains_vs_m3 = aggregate_gains(summary, "M3")
    gains_vs_history = aggregate_gains(summary, "M3_H_raw")
    gains_vs_sd = aggregate_gains(summary, "M3_H_raw_S_D")
    gains_vs_headline = aggregate_gains(summary, "adaptive_two_stage")
    gains_vs_m3.to_csv(run_root / "mean_gains_vs_m3.csv", index=False)
    gains_vs_history.to_csv(run_root / "mean_gains_vs_history.csv", index=False)
    gains_vs_sd.to_csv(run_root / "mean_gains_vs_sd.csv", index=False)
    gains_vs_headline.to_csv(run_root / "mean_gains_vs_adaptive_two_stage.csv", index=False)
    critical_cohort_gains(cohort_frame, "M3_H_raw_S_D").to_csv(run_root / "critical_cohort_uncertainty_gains_vs_sd.csv", index=False)
    critical_cohort_gains(cohort_frame, "adaptive_two_stage").to_csv(run_root / "critical_cohort_uncertainty_gains_vs_headline.csv", index=False)

    panel_models = {"M3_H_raw_S_D_sigma2", "M3_H_raw_S_D_z", "M3_H_raw_S_D_sigma2_z"}
    headline_gains = gains_vs_headline[gains_vs_headline["model"].isin(panel_models)].copy()
    sd_gains = gains_vs_sd[gains_vs_sd["model"].isin(panel_models)].copy()
    accepted_rows = []
    for _, row in headline_gains.iterrows():
        accepted_rows.append(
            {
                "model": row["model"],
                "accepted_vs_current_v2_headline": bool(
                    row["win_count_auc_pr"] >= 4
                    or row["win_count_precision_at_0.01"] >= 4
                    or row["win_count_precision_at_0.005"] >= 4
                ),
                "headline_win_count_auc_pr": int(row["win_count_auc_pr"]),
                "headline_mean_gain_auc_pr": float(row["mean_gain_auc_pr"]),
                "headline_win_count_precision_at_0.01": int(row["win_count_precision_at_0.01"]),
                "headline_mean_gain_precision_at_0.01": float(row["mean_gain_precision_at_0.01"]),
            }
        )
    acceptance = pd.DataFrame(accepted_rows).sort_values(
        ["accepted_vs_current_v2_headline", "headline_mean_gain_precision_at_0.01", "headline_mean_gain_auc_pr"],
        ascending=[False, False, False],
    )
    acceptance.to_csv(run_root / "acceptance_summary.csv", index=False)

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        view = frame[[c for c in columns if c in frame.columns]].copy()
        if view.empty:
            return "_No rows._"

        def fmt(value) -> str:
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6g}"
            return str(value)

        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(fmt(value) for value in row) + " |" for row in view.to_numpy()]
        return "\n".join([header, sep, *rows])

    acceptance_cols = [
        "model",
        "accepted_vs_current_v2_headline",
        "headline_mean_gain_auc_pr",
        "headline_win_count_auc_pr",
        "headline_mean_gain_precision_at_0.01",
        "headline_win_count_precision_at_0.01",
    ]
    gain_cols = [
        "model",
        "baseline",
        "windows",
        "mean_gain_auc_pr",
        "win_count_auc_pr",
        "mean_gain_precision_at_0.005",
        "win_count_precision_at_0.005",
        "mean_gain_precision_at_0.01",
        "win_count_precision_at_0.01",
        "mean_gain_precision_at_0.02",
        "win_count_precision_at_0.02",
        "mean_gain_precision_at_0.05",
        "win_count_precision_at_0.05",
    ]
    total_seconds = sum(float(payload.get("seconds", 0.0)) for payload in runtimes.values())
    feature_seconds = sum(float(payload.get("posterior_runtime", {}).get("feature_seconds", 0.0)) for payload in runtimes.values())
    report = [
        "# Posterior variance / z diagnostic",
        "",
        "Frozen v2 block-causal uncertainty panel. No Random Forest, no new feature grid, no test tuning.",
        "",
        f"Total runtime: {total_seconds:.1f} seconds ({total_seconds / 60:.1f} minutes).",
        f"Posterior feature construction time: {feature_seconds:.1f} seconds ({feature_seconds / 60:.1f} minutes).",
        "",
        "## Acceptance against current v2 headline",
        "",
        table(acceptance, acceptance_cols),
        "",
        "## Mean gains over M3 + H_raw + S_D",
        "",
        table(sd_gains[sd_gains["model"].isin(panel_models)], gain_cols),
        "",
        "## Mean gains over adaptive_two_stage",
        "",
        table(headline_gains, gain_cols),
    ]
    (run_root / "posterior_uncertainty_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/00_moderate_100k_block_causal")
    parser.add_argument("--crossfit-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/00_crossfit_reranker_block_causal")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--precision-alpha", type=float, default=5.0)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--force-features", action="store_true")
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
