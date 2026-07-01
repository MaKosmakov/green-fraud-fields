from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import log1p, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from .ieee_cis import TypedEdge, build_transaction_edges, transaction_edge_records
from .laplacian_features import LayerGraph


@dataclass
class LayerState:
    graph: LayerGraph = field(default_factory=LayerGraph)
    node_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    edge_count: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    node_last: dict[str, float] = field(default_factory=dict)
    edge_last: dict[tuple[str, str], float] = field(default_factory=dict)
    amount_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    amount_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_times: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _safe_prefix(layer: str) -> str:
    return layer.replace("--", "__").replace(" ", "_")


class CausalFeatureBuilder:
    def __init__(
        self,
        lambda_: float = 1.0,
        epsilon: float = 1e-6,
        ego_radius: int = 2,
        max_ego_nodes: int = 250,
        short_window: float = 86400.0,
        long_window: float = 7 * 86400.0,
        compute_laplacian: bool = True,
    ):
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.ego_radius = ego_radius
        self.max_ego_nodes = max_ego_nodes
        self.short_window = short_window
        self.long_window = long_window
        self.compute_laplacian = compute_laplacian
        self.layers: dict[str, LayerState] = defaultdict(LayerState)

    @staticmethod
    def _neighbors(state: LayerState, node: str):
        return state.graph.adjacency.get(node, {}).keys()

    def _edge_features(self, edge: TypedEdge, amount: float) -> dict[str, float | str]:
        state = self.layers[edge.layer]
        a, b, now = edge.a, edge.b, edge.time
        pair = _pair(a, b)
        a_seen, b_seen = a in state.node_last, b in state.node_last
        edge_seen = pair in state.edge_last
        if not a_seen and not b_seen:
            cohort = "C11"
        elif a_seen != b_seen:
            cohort = "C10"
        elif edge_seen:
            cohort = "C00_oldedge"
        else:
            cohort = "C00_newedge"
        na, nb = self._neighbors(state, a), self._neighbors(state, b)
        da, db = len(na), len(nb)
        small, large = (na, nb) if da <= db else (nb, na)
        common = {node for node in small if node in large}
        union_size = da + db - len(common)
        adamic = sum(1.0 / np.log(max(len(self._neighbors(state, n)), 2)) for n in common)
        lap = (
            state.graph.features(
                a,
                b,
                lambda_=self.lambda_,
                eps=self.epsilon,
                radius=self.ego_radius,
                max_nodes=self.max_ego_nodes,
            )
            if self.compute_laplacian else {}
        )
        mean_a = state.amount_sum[a] / state.amount_count[a] if state.amount_count[a] else np.nan
        mean_b = state.amount_sum[b] / state.amount_count[b] if state.amount_count[b] else np.nan
        delta = mean_a - mean_b if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan
        t_edge = (
            delta / sqrt(lap["R_ab"] + self.epsilon)
            if self.compute_laplacian and np.isfinite(delta) else np.nan
        )
        current_log = log1p(max(amount, 0.0))
        ta = ((current_log - mean_a) / sqrt(lap["G_aa"] + self.epsilon)
              if self.compute_laplacian and np.isfinite(mean_a) else np.nan)
        tb = ((current_log - mean_b) / sqrt(lap["G_bb"] + self.epsilon)
              if self.compute_laplacian and np.isfinite(mean_b) else np.nan)
        short_counts, long_counts = [], []
        for node in (a, b):
            history = state.recent_times[node]
            while history and history[0] < now - self.long_window:
                history.popleft()
            long_count = len(history)
            short_count = sum(t >= now - self.short_window for t in history)
            short_counts.append(short_count)
            long_counts.append(long_count)
        velocity = max(
            log1p(s) - log1p(l * self.short_window / self.long_window)
            for s, l in zip(short_counts, long_counts)
        )
        return {
            "a_seen": int(a_seen),
            "b_seen": int(b_seen),
            "edge_seen": int(edge_seen),
            "degree_a": da,
            "degree_b": db,
            "count_a": state.node_count[a],
            "count_b": state.node_count[b],
            "count_ab": state.edge_count[pair],
            "time_since_a": now - state.node_last[a] if a_seen else np.nan,
            "time_since_b": now - state.node_last[b] if b_seen else np.nan,
            "time_since_edge": now - state.edge_last[pair] if edge_seen else np.nan,
            "cohort": cohort,
            "common_neighbors": len(common),
            "jaccard": len(common) / union_size if union_size else 0.0,
            "adamic_adar": adamic,
            "degree_min": min(da, db),
            "degree_max": max(da, db),
            "degree_sum": da + db,
            "degree_product": da * db,
            "preferential_attachment": da * db,
            **lap,
            "amount_mean_a": mean_a,
            "amount_mean_b": mean_b,
            "delta_log_amt_mean": delta,
            "t_edge_amount": t_edge,
            "t_edge_amount_sq": t_edge * t_edge if np.isfinite(t_edge) else np.nan,
            "t_endpoint_a_amount": ta,
            "t_endpoint_b_amount": tb,
            "t_endpoint_max_abs": (
                max(abs(ta), abs(tb)) if np.isfinite(ta) and np.isfinite(tb) else np.nan
            ),
            "t_endpoint_sum_sq": (
                ta * ta + tb * tb if np.isfinite(ta) and np.isfinite(tb) else np.nan
            ),
            "count_velocity": velocity,
        }

    def _insert(self, edge: TypedEdge, amount: float) -> None:
        state = self.layers[edge.layer]
        pair = _pair(edge.a, edge.b)
        logged_amount = log1p(max(amount, 0.0))
        state.graph.add_edge(edge.a, edge.b, edge.w)
        for node in (edge.a, edge.b):
            state.node_count[node] += 1
            state.node_last[node] = edge.time
            state.amount_sum[node] += logged_amount
            state.amount_count[node] += 1
            state.recent_times[node].append(edge.time)
        state.edge_count[pair] += 1
        state.edge_last[pair] = edge.time

    def process_edges(self, edges: Iterable[TypedEdge], amount: float) -> dict[str, object]:
        edges = list(edges)
        per_layer: dict[str, dict[str, object]] = {}
        for edge in edges:
            per_layer[edge.layer] = self._edge_features(edge, amount)
        output: dict[str, object] = {}
        for layer, values in per_layer.items():
            prefix = _safe_prefix(layer)
            output.update({f"{prefix}__{key}": value for key, value in values.items()})
        numeric_keys = sorted(
            {key for values in per_layer.values() for key, value in values.items()
             if key != "cohort" and isinstance(value, (int, float, np.integer, np.floating))}
        )
        for key in numeric_keys:
            values = [float(v[key]) for v in per_layer.values() if key in v and pd.notna(v[key])]
            output[f"agg__{key}__max"] = max(values) if values else np.nan
            output[f"agg__{key}__mean"] = float(np.mean(values)) if values else np.nan
        cohorts = [str(v["cohort"]) for v in per_layer.values()]
        output["cohort_any_C00_newedge"] = int("C00_newedge" in cohorts)
        output["cohort_any_C00_oldedge"] = int("C00_oldedge" in cohorts)
        output["cohort_any_C10"] = int("C10" in cohorts)
        output["cohort_all_C11"] = int(bool(cohorts) and all(c == "C11" for c in cohorts))
        output["known_endpoints_all_edges"] = int(
            bool(per_layer) and all(v["a_seen"] and v["b_seen"] for v in per_layer.values())
        )
        for edge in edges:
            self._insert(edge, amount)
        return output

    def _score_edges_from_frozen_state(self, edges: Iterable[TypedEdge], amount: float) -> tuple[dict[str, object], list[TypedEdge]]:
        edges = list(edges)
        per_layer: dict[str, dict[str, object]] = {}
        for edge in edges:
            per_layer[edge.layer] = self._edge_features(edge, amount)
        output: dict[str, object] = {}
        for layer, values in per_layer.items():
            prefix = _safe_prefix(layer)
            output.update({f"{prefix}__{key}": value for key, value in values.items()})
        numeric_keys = sorted(
            {key for values in per_layer.values() for key, value in values.items()
             if key != "cohort" and isinstance(value, (int, float, np.integer, np.floating))}
        )
        for key in numeric_keys:
            values = [float(v[key]) for v in per_layer.values() if key in v and pd.notna(v[key])]
            output[f"agg__{key}__max"] = max(values) if values else np.nan
            output[f"agg__{key}__mean"] = float(np.mean(values)) if values else np.nan
        cohorts = [str(v["cohort"]) for v in per_layer.values()]
        output["cohort_any_C00_newedge"] = int("C00_newedge" in cohorts)
        output["cohort_any_C00_oldedge"] = int("C00_oldedge" in cohorts)
        output["cohort_any_C10"] = int("C10" in cohorts)
        output["cohort_all_C11"] = int(bool(cohorts) and all(c == "C11" for c in cohorts))
        output["known_endpoints_all_edges"] = int(
            bool(per_layer) and all(v["a_seen"] and v["b_seen"] for v in per_layer.values())
        )
        return output, edges

    def process_timestamp_block(self, block: pd.DataFrame | list[dict]) -> list[dict[str, object]]:
        records = transaction_edge_records(block, include_amount=True) if isinstance(block, pd.DataFrame) else block
        scored: list[dict[str, object]] = []
        pending: list[tuple[list[TypedEdge], float]] = []
        for row in records:
            amount = float(row.get("TransactionAmt", 0.0) or 0.0)
            edges = build_transaction_edges(row)
            output, frozen_edges = self._score_edges_from_frozen_state(edges, amount)
            scored.append(output)
            pending.append((frozen_edges, amount))
        for edges, amount in pending:
            for edge in edges:
                self._insert(edge, amount)
        return scored

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        current_time = None
        block: list[dict] = []
        for row in transaction_edge_records(df, include_amount=True):
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

