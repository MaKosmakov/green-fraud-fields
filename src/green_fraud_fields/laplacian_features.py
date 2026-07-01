from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_spd_dense(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    factor = cho_factor(matrix, lower=False, check_finite=False)
    return cho_solve(factor, rhs, check_finite=False)


@dataclass
class LayerGraph:
    adjacency: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def add_edge(self, a: str, b: str, w: float = 1.0) -> None:
        self.adjacency[a][b] = self.adjacency[a].get(b, 0.0) + w
        self.adjacency[b][a] = self.adjacency[b].get(a, 0.0) + w

    def ego_nodes(self, roots: tuple[str, str], radius: int, max_nodes: int) -> list[str]:
        nodes, _ = self.ego_nodes_bounded(roots, radius, max_nodes)
        return nodes

    def ego_nodes_bounded(
        self, roots: tuple[str, str], radius: int, max_nodes: int
    ) -> tuple[list[str], bool]:
        seen = set(roots)
        queue = deque((root, 0) for root in roots)
        truncated = False
        while queue:
            node, depth = queue.popleft()
            if depth >= radius:
                continue
            for neighbor in self.adjacency.get(node, {}):
                if neighbor not in seen:
                    if len(seen) >= max_nodes:
                        truncated = True
                        return sorted(seen), truncated
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return sorted(seen), truncated

    def features(
        self,
        a: str,
        b: str,
        lambda_: float = 1.0,
        eps: float = 1e-12,
        radius: int = 2,
        max_nodes: int = 250,
    ) -> dict[str, float]:
        nodes, matrix, index = self.ego_system(a, b, lambda_, radius, max_nodes)
        if not self.adjacency.get(a) and not self.adjacency.get(b):
            gaa = gbb = 1.0 / lambda_
            return self._feature_dict(gaa, gbb, 0.0, eps, len(nodes))
        ia, ib = index[a], index[b]
        rhs = np.zeros((len(nodes), 2), dtype=float)
        rhs[ia, 0] = 1.0
        rhs[ib, 1] = 1.0
        if len(nodes) <= 128:
            solution = solve_spd_dense(matrix.toarray(), rhs)
        else:
            solution = np.asarray(spsolve(matrix, rhs))
        gaa = float(solution[ia, 0])
        gbb = float(solution[ib, 1])
        gab = float(0.5 * (solution[ib, 0] + solution[ia, 1]))
        return self._feature_dict(gaa, gbb, gab, eps, len(nodes))

    def ego_system(
        self,
        a: str,
        b: str,
        lambda_: float,
        radius: int,
        max_nodes: int,
    ) -> tuple[list[str], sparse.csc_matrix, dict[str, int]]:
        nodes = self.ego_nodes((a, b), radius=radius, max_nodes=max_nodes)
        matrix, index = self.matrix_for_nodes(nodes, lambda_)
        return nodes, matrix, index

    def matrix_for_nodes(
        self, nodes: list[str], lambda_: float
    ) -> tuple[sparse.csc_matrix, dict[str, int]]:
        matrix, index, _ = self.matrix_for_nodes_timed(nodes, lambda_)
        return matrix, index

    def matrix_for_nodes_timed(
        self, nodes: list[str], lambda_: float
    ) -> tuple[sparse.csc_matrix, dict[str, int], dict[str, float]]:
        """Build the induced-node Laplacian with timing breakdown.

        The important optimization is to avoid scanning a high-degree hub's
        whole adjacency list when the capped ego has only a small number of
        nodes.  For hub-heavy IEEE-CIS layers, iterating over the smaller of
        ``neighbors`` and ``nodes`` preserves the exact induced matrix while
        removing the old matrix-assembly pathology.
        """
        started = time.perf_counter()
        index = {node: i for i, node in enumerate(nodes)}
        indexed_at = time.perf_counter()
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        degree = np.zeros(len(nodes), dtype=float)
        edge_visits = 0
        for u in nodes:
            i = index[u]
            neighbors = self.adjacency.get(u, {})
            if len(neighbors) <= len(nodes):
                iterator = neighbors.items()
            else:
                iterator = ((v, neighbors.get(v, 0.0)) for v in nodes if v != u)
            for v, weight in iterator:
                if weight and v in index:
                    edge_visits += 1
                    degree[i] += weight
                    rows.append(i)
                    cols.append(index[v])
                    values.append(-weight)
        extracted_at = time.perf_counter()
        rows.extend(range(len(nodes)))
        cols.extend(range(len(nodes)))
        values.extend((degree + lambda_).tolist())
        diagonal_at = time.perf_counter()
        matrix = sparse.csc_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes)))
        assembled_at = time.perf_counter()
        timing = {
            "path": "sparse",
            "index_seconds": indexed_at - started,
            "edge_extraction_seconds": extracted_at - indexed_at,
            "diagonal_seconds": diagonal_at - extracted_at,
            "coo_csc_assembly_seconds": assembled_at - diagonal_at,
            "matrix_assembly_seconds": assembled_at - started,
            "edge_visits": float(edge_visits),
            "nnz": float(matrix.nnz),
        }
        return matrix, index, timing

    def dense_matrix_for_nodes_timed(
        self, nodes: list[str], lambda_: float
    ) -> tuple[np.ndarray, dict[str, int], dict[str, float]]:
        """Build the induced-node ``L + lambda I`` matrix as dense float64.

        For the current radius-2/cap-100 experiments the matrix is tiny
        (at most 100x100), so direct dense assembly avoids SciPy sparse
        constructor stalls while preserving the exact same induced graph.
        """
        started = time.perf_counter()
        index = {node: i for i, node in enumerate(nodes)}
        indexed_at = time.perf_counter()
        matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
        allocated_at = time.perf_counter()
        edge_visits = 0
        edge_count = 0
        for u in nodes:
            i = index[u]
            neighbors = self.adjacency.get(u, {})
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
                edge_visits += 1
                edge_count += 1
                matrix[i, i] += weight
                matrix[j, j] += weight
                matrix[i, j] -= weight
                matrix[j, i] -= weight
        extracted_at = time.perf_counter()
        matrix.flat[:: len(nodes) + 1] += lambda_
        diagonal_at = time.perf_counter()
        timing = {
            "path": "dense",
            "index_seconds": indexed_at - started,
            "dense_allocation_seconds": allocated_at - indexed_at,
            "edge_extraction_seconds": extracted_at - allocated_at,
            "diagonal_seconds": diagonal_at - extracted_at,
            "dense_assembly_seconds": diagonal_at - started,
            "matrix_assembly_seconds": diagonal_at - started,
            "edge_visits": float(edge_visits),
            "edge_count": float(edge_count),
            "nnz": float(np.count_nonzero(matrix)),
        }
        return matrix, index, timing

    @staticmethod
    def _feature_dict(
        gaa: float,
        gbb: float,
        gab: float,
        eps: float,
        node_count: int,
    ) -> dict[str, float]:
        resistance = max(gaa + gbb - 2.0 * gab, 0.0)
        return {
            "G_aa": gaa,
            "G_bb": gbb,
            "G_ab": gab,
            "R_ab": resistance,
            "logdet_edge": float(np.log1p(resistance)),
            "fragility_diag_sum": gaa + gbb,
            "fragility_diag_max": max(gaa, gbb),
            "cross_ratio": gab / np.sqrt(max(gaa * gbb, 0.0) + eps),
            "resistance_additive_gap": resistance - gaa - gbb,
            "ego_nodes": float(node_count),
        }


def exact_edge_features(
    weighted_edges: list[tuple[str, str, float]],
    a: str,
    b: str,
    lambda_: float = 1.0,
) -> dict[str, float]:
    graph = LayerGraph()
    for u, v, w in weighted_edges:
        graph.add_edge(u, v, w)
    return graph.features(a, b, lambda_=lambda_, radius=10_000, max_nodes=100_000)

