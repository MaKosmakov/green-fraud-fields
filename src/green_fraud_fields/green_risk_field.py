from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import heapq
from math import log1p
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve

from .ieee_cis import TypedEdge, build_transaction_edges, transaction_edge_records
from .laplacian_features import LayerGraph, solve_spd_dense

RISK_LAYERS = {
    "card1--addr1",
    "card1--P_emaildomain",
    "card1--DeviceInfo",
    "DeviceInfo--P_emaildomain",
}


def select_by_validation(candidates, score_index: int = 0):
    """Select a candidate using validation information only."""
    return max(candidates, key=lambda item: item[score_index])


@dataclass
class ReleasedHistory:
    exposure: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fraud: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    global_exposure: int = 0
    global_fraud: int = 0

    @property
    def global_rate(self) -> float:
        return self.global_fraud / self.global_exposure if self.global_exposure else 0.0

    def update(self, edges: Iterable[TypedEdge], label: int) -> None:
        self.global_exposure += 1
        self.global_fraud += int(label)
        for edge in edges:
            for node in (edge.a, edge.b):
                self.exposure[node] += 1
                self.fraud[node] += int(label)

    def signal(self, node: str, alpha: float) -> float:
        count = self.exposure[node]
        return (self.fraud[node] + alpha * self.global_rate) / (count + alpha)


@dataclass
class DelayedReleaseState:
    delay_seconds: float
    history_by_layer: dict[str, ReleasedHistory] = field(
        default_factory=lambda: defaultdict(ReleasedHistory)
    )
    queue: list[tuple[float, int, int, tuple[TypedEdge, ...]]] = field(default_factory=list)
    sequence: int = 0

    def schedule(self, time: float, label: int, edges: Iterable[TypedEdge]) -> None:
        self.sequence += 1
        heapq.heappush(
            self.queue,
            (time + self.delay_seconds, self.sequence, int(label), tuple(edges)),
        )

    def release_until(self, now: float) -> None:
        while self.queue and self.queue[0][0] <= now:
            _, _, label, edges = heapq.heappop(self.queue)
            by_layer: dict[str, list[TypedEdge]] = defaultdict(list)
            for edge in edges:
                by_layer[edge.layer].append(edge)
            for layer, layer_edges in by_layer.items():
                self.history_by_layer[layer].update(layer_edges, label)

    def release_before(self, now: float) -> None:
        while self.queue and self.queue[0][0] < now:
            _, _, label, edges = heapq.heappop(self.queue)
            by_layer: dict[str, list[TypedEdge]] = defaultdict(list)
            for edge in edges:
                by_layer[edge.layer].append(edge)
            for layer, layer_edges in by_layer.items():
                self.history_by_layer[layer].update(layer_edges, label)


def corrected_transfer(
    shat_a: float,
    shat_b: float,
    gaa: float,
    gbb: float,
    gab: float,
    resistance: float,
    weight: float = 1.0,
) -> tuple[float, float]:
    denominator = 1.0 + weight * resistance
    left = -weight * (shat_a - shat_b) * (gaa - gab) / denominator
    right = -weight * (shat_b - shat_a) * (gbb - gab) / denominator
    return left, right


class GreenRiskFieldBuilder:
    def __init__(
        self,
        delays_days: tuple[int, ...] = (0, 1, 3, 7, 14, 30, 60),
        alphas: tuple[int, ...] = (5, 20, 100),
        lambda_: float = 0.1,
        ego_radius: int = 2,
        max_ego_nodes: int = 100,
        dense_threshold: int = 150,
    ):
        self.delays_days = delays_days
        self.alphas = alphas
        self.lambda_ = lambda_
        self.ego_radius = ego_radius
        self.max_ego_nodes = max_ego_nodes
        self.dense_threshold = dense_threshold
        self.graphs: dict[str, LayerGraph] = defaultdict(LayerGraph)
        self.delays = {
            delay: DelayedReleaseState(delay * 86400.0) for delay in delays_days
        }
        self.ego_sizes: list[int] = []
        self.timing = defaultdict(float)
        self.timing_counts = defaultdict(int)

    def _edge_features(self, edge: TypedEdge) -> dict[str, float]:
        graph = self.graphs[edge.layer]
        t0 = time.perf_counter()
        nodes = graph.ego_nodes((edge.a, edge.b), self.ego_radius, self.max_ego_nodes)
        t1 = time.perf_counter()
        if len(nodes) <= self.dense_threshold:
            matrix, index, matrix_timing = graph.dense_matrix_for_nodes_timed(
                nodes, self.lambda_
            )
            matrix_path = "dense"
        else:
            matrix, index, matrix_timing = graph.matrix_for_nodes_timed(
                nodes, self.lambda_
            )
            matrix_path = "sparse"
        t2 = time.perf_counter()
        for key, value in matrix_timing.items():
            if isinstance(value, (int, float)):
                self.timing[key] += float(value)
        self.timing_counts[f"{matrix_path}_path_count"] += 1
        self.timing["ego_expansion_seconds"] += t1 - t0
        self.timing["matrix_call_seconds"] += t2 - t1
        self.timing_counts["edge_solves"] += 1
        self.ego_sizes.append(len(nodes))
        ia, ib = index[edge.a], index[edge.b]
        signal_keys: list[tuple[int, int]] = []
        rhs_start = time.perf_counter()
        rhs = np.zeros((len(nodes), 2 + len(self.delays_days) * len(self.alphas)))
        rhs[ia, 0] = 1.0
        rhs[ib, 1] = 1.0
        column = 2
        for delay in self.delays_days:
            history = self.delays[delay].history_by_layer[edge.layer]
            for alpha in self.alphas:
                rhs[:, column] = [history.signal(node, alpha) for node in nodes]
                signal_keys.append((delay, alpha))
                column += 1
        rhs_done = time.perf_counter()
        if matrix_path == "dense":
            dense_done = rhs_done
            solution = solve_spd_dense(matrix, rhs)
        else:
            dense = matrix.toarray()
            dense_done = time.perf_counter()
            solution = solve_spd_dense(dense, rhs) if len(nodes) <= 128 else spsolve(matrix, rhs)
        solution = np.asarray(solution)
        solve_done = time.perf_counter()
        self.timing["rhs_signal_seconds"] += rhs_done - rhs_start
        self.timing["dense_conversion_seconds"] += dense_done - rhs_done
        self.timing["linear_solve_seconds"] += solve_done - dense_done
        self.timing[f"{matrix_path}_solve_seconds"] += solve_done - dense_done
        gaa = float(solution[ia, 0])
        gbb = float(solution[ib, 1])
        gab = float(0.5 * (solution[ib, 0] + solution[ia, 1]))
        resistance = max(gaa + gbb - 2 * gab, 0.0)
        output: dict[str, float] = {
            "G_aa": gaa, "G_bb": gbb, "G_ab": gab, "R_ab": resistance,
        }
        for offset, (delay, alpha) in enumerate(signal_keys, start=2):
            history = self.delays[delay].history_by_layer[edge.layer]
            raw_a = history.signal(edge.a, alpha)
            raw_b = history.signal(edge.b, alpha)
            unshrunk_a = (
                history.fraud[edge.a] / history.exposure[edge.a]
                if history.exposure[edge.a] else history.global_rate
            )
            unshrunk_b = (
                history.fraud[edge.b] / history.exposure[edge.b]
                if history.exposure[edge.b] else history.global_rate
            )
            shat_a = self.lambda_ * float(solution[ia, offset])
            shat_b = self.lambda_ * float(solution[ib, offset])
            left_update, right_update = corrected_transfer(
                shat_a, shat_b, gaa, gbb, gab, resistance, edge.w
            )
            prefix = f"d{delay}_a{alpha}"
            output.update({
                f"{prefix}__H_left": raw_a,
                f"{prefix}__H_right": raw_b,
                f"{prefix}__H_raw_left": unshrunk_a,
                f"{prefix}__H_raw_right": unshrunk_b,
                f"{prefix}__H_left_fraud_log": log1p(history.fraud[edge.a]),
                f"{prefix}__H_right_fraud_log": log1p(history.fraud[edge.b]),
                f"{prefix}__H_left_exposure_log": log1p(history.exposure[edge.a]),
                f"{prefix}__H_right_exposure_log": log1p(history.exposure[edge.b]),
                f"{prefix}__H_left_missing": float(history.exposure[edge.a] == 0),
                f"{prefix}__H_right_missing": float(history.exposure[edge.b] == 0),
                f"{prefix}__S_left": shat_a,
                f"{prefix}__S_right": shat_b,
                f"{prefix}__S_sum": shat_a + shat_b,
                f"{prefix}__S_max": max(shat_a, shat_b),
                f"{prefix}__S_min": min(shat_a, shat_b),
                f"{prefix}__S_diff": shat_a - shat_b,
                f"{prefix}__S_absdiff": abs(shat_a - shat_b),
                f"{prefix}__S_left_minus_raw": shat_a - raw_a,
                f"{prefix}__S_right_minus_raw": shat_b - raw_b,
                f"{prefix}__TR_left_update": left_update,
                f"{prefix}__TR_right_update": right_update,
                f"{prefix}__TR_abs_left_update": abs(left_update),
                f"{prefix}__TR_abs_right_update": abs(right_update),
                f"{prefix}__TR_sum_abs_update": abs(left_update) + abs(right_update),
                f"{prefix}__TR_max_abs_update": max(abs(left_update), abs(right_update)),
                f"{prefix}__TR_denominator": 1.0 + edge.w * resistance,
                f"{prefix}__TR_gap": shat_a - shat_b,
                f"{prefix}__TR_gap_abs": abs(shat_a - shat_b),
            })
        return output

    def process_row(self, row: pd.Series) -> dict[str, float]:
        now = float(row["TransactionDT"])
        release_start = time.perf_counter()
        for state in self.delays.values():
            state.release_before(now)
        release_done = time.perf_counter()
        edge_start = time.perf_counter()
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        edge_done = time.perf_counter()
        per_layer = {edge.layer: self._edge_features(edge) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({f"{safe}__{key}": value for key, value in values.items()})
        for delay in self.delays_days:
            for alpha in self.alphas:
                prefix = f"d{delay}_a{alpha}"
                left = [v[f"{prefix}__S_left"] for v in per_layer.values()]
                right = [v[f"{prefix}__S_right"] for v in per_layer.values()]
                absdiff = [v[f"{prefix}__S_absdiff"] for v in per_layer.values()]
                if left:
                    endpoints = left + right
                    output.update({
                        f"agg__{prefix}__S_all_sum": float(np.sum(endpoints)),
                        f"agg__{prefix}__S_all_max": float(np.max(endpoints)),
                        f"agg__{prefix}__S_all_mean": float(np.mean(endpoints)),
                        f"agg__{prefix}__S_all_absdiff_max": float(np.max(absdiff)),
                        f"agg__{prefix}__S_all_absdiff_mean": float(np.mean(absdiff)),
                    })
        graph_start = time.perf_counter()
        for edge in edges:
            self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
        graph_done = time.perf_counter()
        label = int(row["isFraud"])
        for state in self.delays.values():
            state.schedule(now, label, edges)
        schedule_done = time.perf_counter()
        self.timing["release_seconds"] += release_done - release_start
        self.timing["edge_build_seconds"] += edge_done - edge_start
        self.timing["graph_update_seconds"] += graph_done - graph_start
        self.timing["schedule_seconds"] += schedule_done - graph_done
        self.timing_counts["rows"] += 1
        return output

    def _score_row_from_frozen_state(self, row: pd.Series) -> tuple[dict[str, float], list[TypedEdge]]:
        edge_start = time.perf_counter()
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        edge_done = time.perf_counter()
        per_layer = {edge.layer: self._edge_features(edge) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({f"{safe}__{key}": value for key, value in values.items()})
        for delay in self.delays_days:
            for alpha in self.alphas:
                prefix = f"d{delay}_a{alpha}"
                left = [v[f"{prefix}__S_left"] for v in per_layer.values()]
                right = [v[f"{prefix}__S_right"] for v in per_layer.values()]
                absdiff = [v[f"{prefix}__S_absdiff"] for v in per_layer.values()]
                if left:
                    endpoints = left + right
                    output.update({
                        f"agg__{prefix}__S_all_sum": float(np.sum(endpoints)),
                        f"agg__{prefix}__S_all_max": float(np.max(endpoints)),
                        f"agg__{prefix}__S_all_mean": float(np.mean(endpoints)),
                        f"agg__{prefix}__S_all_absdiff_max": float(np.max(absdiff)),
                        f"agg__{prefix}__S_all_absdiff_mean": float(np.mean(absdiff)),
                    })
        self.timing["edge_build_seconds"] += edge_done - edge_start
        return output, edges

    def _commit_block(self, rows_and_edges: list[tuple[dict, list[TypedEdge]]]) -> None:
        graph_start = time.perf_counter()
        for _, edges in rows_and_edges:
            for edge in edges:
                self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
        graph_done = time.perf_counter()
        for row, edges in rows_and_edges:
            now = float(row["TransactionDT"])
            label = int(row["isFraud"])
            for state in self.delays.values():
                state.schedule(now, label, edges)
        schedule_done = time.perf_counter()
        self.timing["graph_update_seconds"] += graph_done - graph_start
        self.timing["schedule_seconds"] += schedule_done - graph_done
        self.timing_counts["rows"] += len(rows_and_edges)

    def process_timestamp_block(self, block: pd.DataFrame | list[dict]) -> list[dict[str, float]]:
        records = transaction_edge_records(block, include_label=True) if isinstance(block, pd.DataFrame) else block
        if not records:
            return []
        now = float(records[0]["TransactionDT"])
        release_start = time.perf_counter()
        for state in self.delays.values():
            state.release_before(now)
        release_done = time.perf_counter()
        scored: list[dict[str, float]] = []
        rows_and_edges: list[tuple[dict, list[TypedEdge]]] = []
        for row in records:
            output, edges = self._score_row_from_frozen_state(row)
            scored.append(output)
            rows_and_edges.append((row, edges))
        self._commit_block(rows_and_edges)
        self.timing["release_seconds"] += release_done - release_start
        return scored

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        current_time = None
        block: list[dict] = []
        for row in transaction_edge_records(df, include_label=True):
            now = float(row["TransactionDT"])
            if current_time is None:
                current_time = now
            if now != current_time:
                rows.extend(self.process_timestamp_block(block))
                block = []
                current_time = now
            block.append(row)
        rows.extend(self.process_timestamp_block(block))
        return pd.DataFrame(rows, index=df.index)

    def write_parquet(
        self,
        df: pd.DataFrame,
        path: str,
        batch_size: int = 500,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

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
                b"delays_days": ",".join(map(str, self.delays_days)).encode(),
                b"ego_radius": str(self.ego_radius).encode(),
                b"max_ego_nodes": str(self.max_ego_nodes).encode(),
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
                    batch_start = time.perf_counter()
                    records = self.process_timestamp_block(block)
                    records_done = time.perf_counter()
                    pending_records.extend(records)
                    if len(pending_records) >= batch_size:
                        write_records(pending_records)
                        pending_records = []
                    write_done = time.perf_counter()
                    self.timing["record_generation_seconds"] += records_done - batch_start
                    self.timing["feature_writing_seconds"] += write_done - records_done
                    block = []
                    current_time = now
                block.append(row)
            if block:
                batch_start = time.perf_counter()
                records = self.process_timestamp_block(block)
                records_done = time.perf_counter()
                pending_records.extend(records)
                if len(pending_records) >= batch_size:
                    write_records(pending_records)
                    pending_records = []
                write_done = time.perf_counter()
                self.timing["record_generation_seconds"] += records_done - batch_start
                self.timing["feature_writing_seconds"] += write_done - records_done
            write_records(pending_records)
        finally:
            if writer is not None:
                writer.close()

    def runtime_summary(self) -> dict[str, float | int]:
        edge_solves = int(self.timing_counts.get("edge_solves", 0))
        rows = int(self.timing_counts.get("rows", 0))
        summary: dict[str, float | int] = {
            "rows": rows,
            "edge_solves": edge_solves,
            "dense_path_count": int(self.timing_counts.get("dense_path_count", 0)),
            "sparse_path_count": int(self.timing_counts.get("sparse_path_count", 0)),
            "mean_ego_size": float(np.mean(self.ego_sizes)) if self.ego_sizes else 0.0,
            "max_ego_size": int(max(self.ego_sizes)) if self.ego_sizes else 0,
        }
        summary.update({key: float(value) for key, value in self.timing.items()})
        if edge_solves:
            summary["mean_matrix_assembly_seconds"] = (
                float(self.timing["matrix_assembly_seconds"]) / edge_solves
            )
            summary["mean_linear_solve_seconds"] = (
                float(self.timing["linear_solve_seconds"]) / edge_solves
            )
        return summary


class BoundedGreenRiskFieldBuilder:
    """Runtime-bounded one-delay Green field with explicit fallback flags."""

    def __init__(
        self,
        delay_days: int = 1,
        alpha: int = 5,
        lambda_: float = 0.1,
        exact_radius: int = 2,
        exact_cap: int = 100,
        fallback_radius: int = 1,
        fallback_cap: int = 30,
        dense_threshold: int = 150,
    ):
        self.delay_days = delay_days
        self.alpha = alpha
        self.lambda_ = lambda_
        self.exact_radius = exact_radius
        self.exact_cap = exact_cap
        self.fallback_radius = fallback_radius
        self.fallback_cap = fallback_cap
        self.dense_threshold = dense_threshold
        self.graphs: dict[str, LayerGraph] = defaultdict(LayerGraph)
        self.release = DelayedReleaseState(delay_days * 86400.0)
        self.counts = defaultdict(int)
        self.ego_sizes: list[int] = []

    def _neighbor_average(self, graph: LayerGraph, history: ReleasedHistory, node: str) -> float:
        neighbors = graph.adjacency.get(node, {})
        if not neighbors:
            return history.signal(node, self.alpha)
        total_weight = sum(neighbors.values())
        if total_weight <= 0:
            return history.signal(node, self.alpha)
        return sum(
            weight * history.signal(neighbor, self.alpha)
            for neighbor, weight in neighbors.items()
        ) / total_weight

    def _solve_field(
        self,
        graph: LayerGraph,
        history: ReleasedHistory,
        a: str,
        b: str,
        radius: int,
        cap: int,
    ) -> tuple[float, float, int, bool]:
        nodes, truncated = graph.ego_nodes_bounded((a, b), radius=radius, max_nodes=cap)
        if len(nodes) <= self.dense_threshold:
            matrix, index, _ = graph.dense_matrix_for_nodes_timed(nodes, self.lambda_)
        else:
            sparse_matrix, index = graph.matrix_for_nodes(nodes, self.lambda_)
            matrix = sparse_matrix.toarray()
        signal = np.array([history.signal(node, self.alpha) for node in nodes], dtype=float)
        solution = solve_spd_dense(matrix, signal)
        field = self.lambda_ * solution
        return float(field[index[a]]), float(field[index[b]]), len(nodes), truncated

    def _edge_features(self, edge: TypedEdge) -> dict[str, float]:
        graph = self.graphs[edge.layer]
        history = self.release.history_by_layer[edge.layer]
        prefix = f"d{self.delay_days}_a{self.alpha}"
        raw_a = history.signal(edge.a, self.alpha)
        raw_b = history.signal(edge.b, self.alpha)
        output = {
            f"{prefix}__H_left": raw_a,
            f"{prefix}__H_right": raw_b,
            f"{prefix}__H_left_exposure_log": log1p(history.exposure[edge.a]),
            f"{prefix}__H_right_exposure_log": log1p(history.exposure[edge.b]),
            f"{prefix}__H_left_fraud_log": log1p(history.fraud[edge.a]),
            f"{prefix}__H_right_fraud_log": log1p(history.fraud[edge.b]),
            f"{prefix}__H_left_missing": float(history.exposure[edge.a] == 0),
            f"{prefix}__H_right_missing": float(history.exposure[edge.b] == 0),
        }
        exact_nodes, exact_truncated = graph.ego_nodes_bounded(
            (edge.a, edge.b), self.exact_radius, self.exact_cap
        )
        if not exact_truncated:
            shat_a, shat_b, ego_size, _ = self._solve_field(
                graph, history, edge.a, edge.b, self.exact_radius, self.exact_cap
            )
            mode = "exact_r2"
        else:
            r1_nodes, r1_truncated = graph.ego_nodes_bounded(
                (edge.a, edge.b), self.fallback_radius, self.fallback_cap
            )
            if not r1_truncated:
                shat_a, shat_b, ego_size, _ = self._solve_field(
                    graph, history, edge.a, edge.b, self.fallback_radius, self.fallback_cap
                )
                mode = "r1_fallback"
            else:
                shat_a = self._neighbor_average(graph, history, edge.a)
                shat_b = self._neighbor_average(graph, history, edge.b)
                ego_size = len(r1_nodes)
                mode = "neighbor_fallback"
        self.counts[mode] += 1
        self.ego_sizes.append(ego_size)
        output.update({
            f"{prefix}__S_left": shat_a,
            f"{prefix}__S_right": shat_b,
            f"{prefix}__S_sum": shat_a + shat_b,
            f"{prefix}__S_max": max(shat_a, shat_b),
            f"{prefix}__S_min": min(shat_a, shat_b),
            f"{prefix}__S_diff": shat_a - shat_b,
            f"{prefix}__S_absdiff": abs(shat_a - shat_b),
            f"{prefix}__S_left_minus_raw": shat_a - raw_a,
            f"{prefix}__S_right_minus_raw": shat_b - raw_b,
            f"{prefix}__S_used_exact_r2": float(mode == "exact_r2"),
            f"{prefix}__S_used_r1_fallback": float(mode == "r1_fallback"),
            f"{prefix}__S_used_neighbor_fallback": float(mode == "neighbor_fallback"),
            f"{prefix}__S_ego_over_cap": float(mode != "exact_r2"),
            f"{prefix}__S_ego_size": float(ego_size),
        })
        return output

    def _score_row_from_frozen_state(self, row: pd.Series) -> tuple[dict[str, float], list]:
        now = float(row["TransactionDT"])
        self.release.release_before(now)
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        per_layer = {edge.layer: self._edge_features(edge) for edge in edges}
        output: dict[str, float] = {}
        prefix = f"d{self.delay_days}_a{self.alpha}"
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({f"{safe}__{key}": value for key, value in values.items()})
        left = [v[f"{prefix}__S_left"] for v in per_layer.values()]
        right = [v[f"{prefix}__S_right"] for v in per_layer.values()]
        absdiff = [v[f"{prefix}__S_absdiff"] for v in per_layer.values()]
        if left:
            endpoints = left + right
            output.update({
                f"agg__{prefix}__S_all_sum": float(np.sum(endpoints)),
                f"agg__{prefix}__S_all_max": float(np.max(endpoints)),
                f"agg__{prefix}__S_all_mean": float(np.mean(endpoints)),
                f"agg__{prefix}__S_all_absdiff_max": float(np.max(absdiff)),
                f"agg__{prefix}__S_all_absdiff_mean": float(np.mean(absdiff)),
                f"agg__{prefix}__S_used_exact_r2": float(max(v[f"{prefix}__S_used_exact_r2"] for v in per_layer.values())),
                f"agg__{prefix}__S_used_r1_fallback": float(max(v[f"{prefix}__S_used_r1_fallback"] for v in per_layer.values())),
                f"agg__{prefix}__S_used_neighbor_fallback": float(max(v[f"{prefix}__S_used_neighbor_fallback"] for v in per_layer.values())),
                f"agg__{prefix}__S_ego_over_cap": float(max(v[f"{prefix}__S_ego_over_cap"] for v in per_layer.values())),
            })
        return output, edges

    def _commit_row(self, row: pd.Series, edges: list) -> None:
        for edge in edges:
            self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
        now = float(row["TransactionDT"])
        self.release.schedule(now, int(row["isFraud"]), edges)

    def process_row(self, row: pd.Series) -> dict[str, float]:
        output, edges = self._score_row_from_frozen_state(row)
        self._commit_row(row, edges)
        return output

    def process_timestamp_block(self, block: pd.DataFrame) -> list[dict[str, float]]:
        if block.empty:
            return []
        now = float(block.iloc[0]["TransactionDT"])
        self.release.release_before(now)
        pending = []
        rows = []
        for _, row in block.iterrows():
            output, edges = self._score_row_from_frozen_state(row)
            rows.append(output)
            pending.append((row, edges))
        for row, edges in pending:
            self._commit_row(row, edges)
        return rows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for _, block in df.groupby("TransactionDT", sort=False):
            rows.extend(self.process_timestamp_block(block))
        return pd.DataFrame(rows, index=df.index)

