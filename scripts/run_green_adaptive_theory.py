from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from math import log1p, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.green_risk_field import DelayedReleaseState, RISK_LAYERS
from green_fraud_fields.ieee_cis import build_transaction_edges, chronological_split, load_ieee_cis, transaction_edge_records
from green_fraud_fields.laplacian_features import LayerGraph, solve_spd_dense
from green_fraud_fields.modeling import evaluate, save_json
from run_green_focused_improvements import (
    fit_named_models,
    selection_key,
    tuned_soft_mixture,
    tuned_two_stage,
)
from run_green_risk_field import base_groups, cohort_metrics, green_columns


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def dense_laplacian(graph: LayerGraph, nodes: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for u in nodes:
        i = index[u]
        neighbors = graph.adjacency.get(u, {})
        if len(neighbors) <= len(nodes):
            iterator = neighbors.items()
        else:
            iterator = ((v, neighbors.get(v, 0.0)) for v in nodes if v != u)
        for v, weight in iterator:
            if not weight or v not in index:
                continue
            j = index[v]
            if j <= i:
                continue
            matrix[i, i] += weight
            matrix[j, j] += weight
            matrix[i, j] -= weight
            matrix[j, i] -= weight
    return matrix, index


class PrecisionWeightedGreenBuilder:
    def __init__(
        self,
        delay_days: int = 0,
        history_alpha: int = 5,
        precision_alpha: float = 5.0,
        radius: int = 2,
        cap: int = 100,
    ):
        self.delay_days = delay_days
        self.history_alpha = history_alpha
        self.precision_alpha = precision_alpha
        self.radius = radius
        self.cap = cap
        self.release = DelayedReleaseState(delay_days * 86400.0)
        self.graphs: dict[str, LayerGraph] = defaultdict(LayerGraph)
        self.counts = defaultdict(int)
        self.ego_sizes: list[int] = []

    def _solve_edge(self, edge) -> dict[str, float]:
        graph = self.graphs[edge.layer]
        history = self.release.history_by_layer[edge.layer]
        nodes = graph.ego_nodes((edge.a, edge.b), radius=self.radius, max_nodes=self.cap)
        laplacian, index = dense_laplacian(graph, nodes)
        h = np.array([history.signal(node, self.history_alpha) for node in nodes], dtype=np.float64)
        exposure = np.array([history.exposure[node] for node in nodes], dtype=np.float64)
        outputs: dict[str, float] = {
            "H_left": history.signal(edge.a, self.history_alpha),
            "H_right": history.signal(edge.b, self.history_alpha),
            "H_left_exposure_log": log1p(history.exposure[edge.a]),
            "H_right_exposure_log": log1p(history.exposure[edge.b]),
        }
        for name, d in {
            "count": self.precision_alpha + exposure,
            "sqrt": self.precision_alpha + np.sqrt(exposure),
        }.items():
            system = laplacian.copy()
            system.flat[:: len(nodes) + 1] += d
            rhs = d * h
            solution = solve_spd_dense(system, rhs)
            left = float(solution[index[edge.a]])
            right = float(solution[index[edge.b]])
            prefix = f"SD_{name}"
            outputs.update({
                f"{prefix}_left": left,
                f"{prefix}_right": right,
                f"{prefix}_sum": left + right,
                f"{prefix}_max": max(left, right),
                f"{prefix}_min": min(left, right),
                f"{prefix}_diff": left - right,
                f"{prefix}_absdiff": abs(left - right),
            })
        self.ego_sizes.append(len(nodes))
        self.counts["edge_solves"] += 1
        return outputs

    def process_row(self, row: pd.Series) -> dict[str, float]:
        now = float(row["TransactionDT"])
        self.release.release_before(now)
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        per_layer = {edge.layer: self._solve_edge(edge) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({f"{safe}__d{self.delay_days}_a{self.history_alpha}__{k}": v for k, v in values.items()})
        for rule in ("count", "sqrt"):
            left = [v[f"SD_{rule}_left"] for v in per_layer.values()]
            right = [v[f"SD_{rule}_right"] for v in per_layer.values()]
            absdiff = [v[f"SD_{rule}_absdiff"] for v in per_layer.values()]
            if left:
                endpoints = left + right
                output.update({
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_sum": float(np.sum(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_max": float(np.max(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_mean": float(np.mean(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_absdiff_max": float(np.max(absdiff)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_absdiff_mean": float(np.mean(absdiff)),
                })
        for edge in edges:
            self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
        self.release.schedule(now, int(row["isFraud"]), edges)
        return output

    def _score_row_from_frozen_state(self, row: pd.Series) -> tuple[dict[str, float], list]:
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        per_layer = {edge.layer: self._solve_edge(edge) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({f"{safe}__d{self.delay_days}_a{self.history_alpha}__{k}": v for k, v in values.items()})
        for rule in ("count", "sqrt"):
            left = [v[f"SD_{rule}_left"] for v in per_layer.values()]
            right = [v[f"SD_{rule}_right"] for v in per_layer.values()]
            absdiff = [v[f"SD_{rule}_absdiff"] for v in per_layer.values()]
            if left:
                endpoints = left + right
                output.update({
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_sum": float(np.sum(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_max": float(np.max(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_mean": float(np.mean(endpoints)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_absdiff_max": float(np.max(absdiff)),
                    f"agg__d{self.delay_days}_a{self.history_alpha}__SD_{rule}_all_absdiff_mean": float(np.mean(absdiff)),
                })
        return output, edges

    def process_timestamp_block(self, block: pd.DataFrame | list[dict]) -> list[dict[str, float]]:
        records = transaction_edge_records(block, include_label=True) if isinstance(block, pd.DataFrame) else block
        if not records:
            return []
        now = float(records[0]["TransactionDT"])
        self.release.release_before(now)
        scored: list[dict[str, float]] = []
        pending: list[tuple[dict, list]] = []
        for row in records:
            output, edges = self._score_row_from_frozen_state(row)
            scored.append(output)
            pending.append((row, edges))
        for row, edges in pending:
            for edge in edges:
                self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
            self.release.schedule(float(row["TransactionDT"]), int(row["isFraud"]), edges)
        return scored

    def write_parquet(self, df: pd.DataFrame, path: Path, batch_size: int = 500) -> dict:
        import pyarrow as pa
        import pyarrow.parquet as pq

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
            metadata.update({
                b"causal_policy": b"strict_timestamp_block",
                b"same_timestamp_policy": b"block_frozen_t_minus",
                b"label_release_policy": b"release_time_strictly_less_than_candidate_timestamp",
                b"delay_days": str(self.delay_days).encode(),
                b"history_alpha": str(self.history_alpha).encode(),
                b"precision_alpha": str(self.precision_alpha).encode(),
                b"ego_radius": str(self.radius).encode(),
                b"max_ego_nodes": str(self.cap).encode(),
                b"cache_version": b"block_causal_v1",
            })
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
            if len(pending_records) >= batch_size:
                write_records(pending_records)
                pending_records = []
            write_records(pending_records)
        finally:
            if writer is not None:
                writer.close()
        return {
            "feature_seconds": time.time() - started,
            "edge_solves": int(self.counts["edge_solves"]),
            "mean_ego_size": float(np.mean(self.ego_sizes)) if self.ego_sizes else 0.0,
            "max_ego_size": int(max(self.ego_sizes)) if self.ego_sizes else 0,
        }


def add_adaptive_shrinkage(green: pd.DataFrame, delays: tuple[int, ...], alpha: int) -> tuple[pd.DataFrame, list[str]]:
    new_cols: dict[str, pd.Series] = {}
    for delay in delays:
        marker = f"d{delay}_a{alpha}__"
        for h_left in [c for c in green.columns if marker in c and c.endswith("__H_left")]:
            base = h_left[:-len("__H_left")]
            h_right = base + "__H_right"
            s_left = base + "__S_left"
            s_right = base + "__S_right"
            e_left = base + "__H_left_exposure_log"
            e_right = base + "__H_right_exposure_log"
            if not all(c in green.columns for c in (h_right, s_left, s_right, e_left, e_right)):
                continue
            n_left = np.expm1(green[e_left].astype(float)).clip(lower=0)
            n_right = np.expm1(green[e_right].astype(float)).clip(lower=0)
            for rule, gamma_left, gamma_right in (
                ("count", 1.0 / (1.0 + n_left), 1.0 / (1.0 + n_right)),
                ("sqrt", 1.0 / np.sqrt(1.0 + n_left), 1.0 / np.sqrt(1.0 + n_right)),
            ):
                left = green[h_left] + gamma_left * (green[s_left] - green[h_left])
                right = green[h_right] + gamma_right * (green[s_right] - green[h_right])
                prefix = base + f"__SG_{rule}"
                new_cols[prefix + "_left"] = left.astype("float32")
                new_cols[prefix + "_right"] = right.astype("float32")
                new_cols[prefix + "_sum"] = (left + right).astype("float32")
                new_cols[prefix + "_max"] = np.maximum(left, right).astype("float32")
                new_cols[prefix + "_absdiff"] = np.abs(left - right).astype("float32")
    if not new_cols:
        return green, []
    frame = pd.DataFrame(new_cols, index=green.index)
    return pd.concat([green, frame], axis=1), list(frame.columns)


def cohort_gate(
    base: pd.DataFrame,
    y_valid: np.ndarray,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    valid: slice,
    test: slice,
    candidates: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_frame = base.iloc[valid].reset_index(drop=True)
    test_frame = base.iloc[test].reset_index(drop=True)
    masks_valid = {
        "C00_newedge": valid_frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": (
            valid_frame["known_endpoints_all_edges"].to_numpy(bool)
            & ~valid_frame["cohort_any_C00_newedge"].to_numpy(bool)
        ),
    }
    masks_valid["other"] = ~(masks_valid["C00_newedge"] | masks_valid["known_endpoints"])
    masks_test = {
        "C00_newedge": test_frame["cohort_any_C00_newedge"].to_numpy(bool),
        "known_endpoints": (
            test_frame["known_endpoints_all_edges"].to_numpy(bool)
            & ~test_frame["cohort_any_C00_newedge"].to_numpy(bool)
        ),
    }
    masks_test["other"] = ~(masks_test["C00_newedge"] | masks_test["known_endpoints"])
    valid_score = np.zeros(len(valid_frame))
    test_score = np.zeros(len(test_frame))
    chosen = {}
    for cohort, mask in masks_valid.items():
        if mask.sum() >= 20 and np.unique(y_valid[mask]).size == 2:
            chosen_model = max(candidates, key=lambda name: selection_key(y_valid[mask], valid_predictions[name][mask]))
        else:
            chosen_model = max(candidates, key=lambda name: selection_key(y_valid, valid_predictions[name]))
        chosen[cohort] = chosen_model
        valid_score[mask] = valid_predictions[chosen_model][mask]
        test_score[masks_test[cohort]] = test_predictions[chosen_model][masks_test[cohort]]
    return valid_score, test_score, {"chosen_models": chosen, "test_used_for_selection": False}


def run_window(args, window: int) -> dict:
    started = time.time()
    baseline_dir = Path(args.baseline_dir) / f"window_{window}"
    out = Path(args.out_dir) / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    data = load_ieee_cis(args.data_dir, (window + 1) * args.window_size)
    data = data.iloc[window * args.window_size:(window + 1) * args.window_size].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    base = pd.read_parquet(baseline_dir / "base_features.parquet").reset_index(drop=True)
    delays = parse_ints(args.delays)
    alpha = args.alpha
    green_path = baseline_dir / "green_features.parquet"
    schema = pq.ParquetFile(green_path).schema.names
    hcols = [c for d in delays for c in green_columns(schema, d, alpha, "H")]
    scols = [c for d in delays for c in green_columns(schema, d, alpha, "S")]
    green = pd.read_parquet(green_path, columns=hcols + scols)
    green, sg_cols = add_adaptive_shrinkage(green, delays, alpha)
    sd_path = out / "precision_weighted_green.parquet"
    if sd_path.exists() and not args.force_sd:
        sd_runtime = {"cached": True}
    else:
        builder = PrecisionWeightedGreenBuilder(delay_days=0, history_alpha=alpha, precision_alpha=5.0)
        sd_runtime = builder.write_parquet(data, sd_path)
        sd_runtime["cached"] = False
    sd = pd.read_parquet(sd_path)
    sd_cols = [c for c in sd.columns if "__SD_" in c or "__SD" in c]
    sd = sd[sd_cols]
    frame = pd.concat([base, green, sd], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    baseline = base_groups(list(base.columns))["BP"]
    model_cols = {
        "M3": baseline,
        "H_S": baseline + hcols + scols,
        "adaptive_shrinkage": baseline + hcols + sg_cols,
        "precision_green_count": baseline + hcols + [c for c in sd_cols if "SD_count" in c],
        "precision_green_sqrt": baseline + hcols + [c for c in sd_cols if "SD_sqrt" in c],
    }
    valid_predictions, test_predictions, selection = fit_named_models(frame, y, train, valid, test, model_cols, args.seed)
    mix_valid, mix_test, mix_selection = tuned_soft_mixture(
        y[valid], valid_predictions, test_predictions,
        ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt"],
    )
    valid_predictions["adaptive_soft_mixture"] = mix_valid
    test_predictions["adaptive_soft_mixture"] = mix_test
    selection["adaptive_soft_mixture"] = mix_selection
    gate_valid, gate_test, gate_selection = cohort_gate(
        base, y[valid], valid_predictions, test_predictions, valid, test,
        ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt", "adaptive_soft_mixture"],
    )
    valid_predictions["cohort_gate_adaptive"] = gate_valid
    test_predictions["cohort_gate_adaptive"] = gate_test
    selection["cohort_gate_adaptive"] = gate_selection
    summary_cols = [c for c in sg_cols + sd_cols if c.startswith("agg__") or "all_" in c]
    two_valid, two_test, two_selection = tuned_two_stage(
        frame, y, valid, test, valid_predictions, test_predictions,
        ["M3", "H_S", "adaptive_shrinkage", "precision_green_count", "precision_green_sqrt", "adaptive_soft_mixture"],
        summary_cols,
    )
    valid_predictions["adaptive_two_stage"] = two_valid
    test_predictions["adaptive_two_stage"] = two_test
    selection["adaptive_two_stage"] = two_selection
    report_models = list(valid_predictions)
    metrics = {
        name: {"validation": evaluate(y[valid], valid_predictions[name]), "test": evaluate(y[test], test_predictions[name])}
        for name in report_models
    }
    cohorts = cohort_metrics(base.iloc[test].reset_index(drop=True), y[test], {name: test_predictions[name] for name in report_models})
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    runtime = {
        "seconds": time.time() - started,
        "window": window,
        "sd_runtime": sd_runtime,
        "uses_cached_baseline_green": True,
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
    }
    save_json(out / "runtime.json", runtime)
    rows = []
    for name in report_models:
        test_metrics = metrics[name]["test"]
        rows.append({
            "window": window,
            "model": name,
            "auc_pr": test_metrics["auc_pr"],
            "precision_at_0.005": test_metrics["precision_at_0.005"],
            "precision_at_0.01": test_metrics["precision_at_0.01"],
            "precision_at_0.02": test_metrics["precision_at_0.02"],
            "precision_at_0.05": test_metrics["precision_at_0.05"],
            "runtime_seconds": runtime["seconds"],
            "sd_feature_seconds": sd_runtime.get("feature_seconds"),
        })
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({row["model"]: row for row in rows}, indent=2))
    return {"summary": rows, "cohorts": cohorts, "runtime": runtime}


def write_aggregate_outputs(root: Path, summaries: list[dict] | None = None, cohorts: list[dict] | None = None, runtimes: dict | None = None) -> None:
    if summaries is None:
        summaries = []
        for path in sorted(root.glob("window_*/summary.csv")):
            summaries.extend(pd.read_csv(path).to_dict("records"))
    if cohorts is None:
        cohorts = []
        for path in sorted(root.glob("window_*/cohort_metrics.json")):
            window = int(path.parent.name.split("_")[-1])
            rows = json.loads(path.read_text())["rows"]
            for row in rows:
                row["window"] = window
            cohorts.extend(rows)
    if runtimes is None:
        runtimes = {}
        for path in sorted(root.glob("window_*/runtime.json")):
            runtimes[path.parent.name] = json.loads(path.read_text())

    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    summary.to_csv(root / "window_summary.csv", index=False)
    cohort_frame.to_csv(root / "cohort_metrics.csv", index=False)
    save_json(root / "runtime.json", runtimes)

    baseline = summary[summary["model"] == "M3"].set_index("window")
    gain_rows = []
    for model, rows in summary.groupby("model"):
        if model == "M3":
            continue
        rows = rows.set_index("window").sort_index()
        common = rows.index.intersection(baseline.index)
        row = {"model": model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            gain = rows.loc[common, metric] - baseline.loc[common, metric]
            row[f"mean_gain_{metric}"] = float(gain.mean())
            row[f"win_count_{metric}"] = int((gain > 0).sum())
        gain_rows.append(row)
    pd.DataFrame(gain_rows).sort_values("mean_gain_precision_at_0.01", ascending=False).to_csv(root / "mean_gains.csv", index=False)

    critical = cohort_frame[cohort_frame["cohort"].isin(["known_endpoints", "C00_newedge"])]
    critical_rows = []
    for (cohort, model), rows in critical.groupby(["cohort", "model"]):
        if model == "M3":
            continue
        base_rows = critical[(critical["cohort"] == cohort) & (critical["model"] == "M3")].set_index("window")
        rows = rows.set_index("window").sort_index()
        common = rows.index.intersection(base_rows.index)
        out = {"cohort": cohort, "model": model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            gain = rows.loc[common, metric] - base_rows.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(gain.mean())
            out[f"win_count_{metric}"] = int((gain > 0).sum())
        critical_rows.append(out)
    pd.DataFrame(critical_rows).sort_values(["cohort", "mean_gain_precision_at_0.01"], ascending=[True, False]).to_csv(
        root / "critical_cohort_mean_gains.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--out-dir", default="outputs/ieee_green_adaptive_theory_v1")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-sd", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        write_aggregate_outputs(root)
        return
    summaries, cohorts, runtimes = [], [], {}
    for window in parse_ints(args.windows):
        result = run_window(args, window)
        summaries.extend(result["summary"])
        runtimes[f"window_{window}"] = result["runtime"]
        for row in result["cohorts"]:
            row["window"] = window
        cohorts.extend(result["cohorts"])
    write_aggregate_outputs(root, summaries, cohorts, runtimes)


if __name__ == "__main__":
    main()

