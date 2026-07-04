from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from math import log1p
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from green_fraud_fields.green_risk_field import RISK_LAYERS, ReleasedHistory
from green_fraud_fields.ieee_cis import build_transaction_edges, chronological_split, load_ieee_cis, load_ieee_cis_cached
from green_fraud_fields.laplacian_features import LayerGraph, solve_spd_dense
from green_fraud_fields.modeling import evaluate, make_preprocessor, save_json
from run_green_adaptive_theory import add_adaptive_shrinkage, dense_laplacian
from run_green_focused_improvements import selection_key, tuned_soft_mixture
from run_green_risk_field import base_groups, cohort_metrics, green_columns
from run_green_risk_tail import rerank
from run_ieee_tail_specialized import fit_tail_selected


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class ExactConfig:
    rule: str
    precision_alpha: float
    lambda_: float

    @property
    def key(self) -> str:
        alpha = str(int(self.precision_alpha)) if self.precision_alpha.is_integer() else str(self.precision_alpha).replace(".", "p")
        lam = str(self.lambda_).replace(".", "p")
        return f"{self.rule}_pa{alpha}_l{lam}"


def parse_configs(value: str) -> tuple[ExactConfig, ...]:
    if value.strip().lower() in {"", "none", "skip"}:
        return ()
    configs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        rule, alpha, lambda_ = item.split(":")
        configs.append(ExactConfig(rule, float(alpha), float(lambda_)))
    return tuple(configs)


@dataclass
class ReleasedRecentState:
    delay_seconds: float
    recent_seconds: float
    history_by_layer: dict[str, ReleasedHistory] = field(default_factory=lambda: defaultdict(ReleasedHistory))
    recent_by_layer: dict[str, dict[str, deque[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(deque))
    )
    queue: list[tuple[float, int, int, tuple]] = field(default_factory=list)
    sequence: int = 0

    def schedule(self, now: float, label: int, edges) -> None:
        self.sequence += 1
        heapq.heappush(self.queue, (now + self.delay_seconds, self.sequence, int(label), tuple(edges)))

    def release_until(self, now: float) -> None:
        while self.queue and self.queue[0][0] <= now:
            release_time, _, label, edges = heapq.heappop(self.queue)
            by_layer: dict[str, list] = defaultdict(list)
            for edge in edges:
                by_layer[edge.layer].append(edge)
            for layer, layer_edges in by_layer.items():
                self.history_by_layer[layer].update(layer_edges, label)
                recent = self.recent_by_layer[layer]
                for edge in layer_edges:
                    recent[edge.a].append(release_time)
                    recent[edge.b].append(release_time)

    def release_before(self, now: float) -> None:
        while self.queue and self.queue[0][0] < now:
            release_time, _, label, edges = heapq.heappop(self.queue)
            by_layer: dict[str, list] = defaultdict(list)
            for edge in edges:
                by_layer[edge.layer].append(edge)
            for layer, layer_edges in by_layer.items():
                self.history_by_layer[layer].update(layer_edges, label)
                recent = self.recent_by_layer[layer]
                for edge in layer_edges:
                    recent[edge.a].append(release_time)
                    recent[edge.b].append(release_time)

    def recent_count(self, layer: str, node: str, now: float) -> int:
        cutoff = now - self.recent_seconds
        queue = self.recent_by_layer[layer][node]
        while queue and queue[0] < cutoff:
            queue.popleft()
        return len(queue)


class ExactAdaptivePanelBuilder:
    def __init__(
        self,
        configs: tuple[ExactConfig, ...],
        delay_days: int = 0,
        history_alpha: int = 5,
        recent_days: int = 7,
        radius: int = 2,
        cap: int = 100,
    ) -> None:
        self.configs = configs
        self.delay_days = delay_days
        self.history_alpha = history_alpha
        self.radius = radius
        self.cap = cap
        self.release = ReleasedRecentState(delay_days * 86400.0, recent_days * 86400.0)
        self.graphs: dict[str, LayerGraph] = defaultdict(LayerGraph)
        self.counts = defaultdict(int)
        self.ego_sizes: list[int] = []

    def expected_columns(self) -> list[str]:
        columns = []
        for layer in sorted(RISK_LAYERS):
            safe = layer.replace("--", "__")
            for config in self.configs:
                prefix = f"{safe}__d{self.delay_days}_ha{self.history_alpha}__SD_{config.key}"
                for suffix in ("left", "right", "sum", "max", "min", "diff", "absdiff"):
                    columns.append(f"{prefix}_{suffix}")
        for config in self.configs:
            prefix = f"agg__d{self.delay_days}_ha{self.history_alpha}__SD_{config.key}"
            for suffix in ("all_sum", "all_max", "all_mean", "all_absdiff_max", "all_absdiff_mean"):
                columns.append(f"{prefix}_{suffix}")
        return columns

    def _precision(self, rule: str, exposure: np.ndarray, recent: np.ndarray, alpha: float) -> np.ndarray:
        if rule == "count":
            return alpha + exposure
        if rule == "sqrt":
            return alpha + np.sqrt(exposure)
        if rule == "log1p":
            return alpha + np.log1p(exposure)
        if rule == "recent":
            return alpha + recent
        if rule == "sqrt_recent":
            return alpha + np.sqrt(recent)
        raise ValueError(f"unknown precision rule: {rule}")

    def _solve_edge(self, edge, now: float) -> dict[str, float]:
        graph = self.graphs[edge.layer]
        history = self.release.history_by_layer[edge.layer]
        nodes = graph.ego_nodes((edge.a, edge.b), radius=self.radius, max_nodes=self.cap)
        laplacian, index = dense_laplacian(graph, nodes)
        h = np.array([history.signal(node, self.history_alpha) for node in nodes], dtype=np.float64)
        exposure = np.array([history.exposure[node] for node in nodes], dtype=np.float64)
        recent = np.array([self.release.recent_count(edge.layer, node, now) for node in nodes], dtype=np.float64)
        outputs: dict[str, float] = {}
        for config in self.configs:
            d = self._precision(config.rule, exposure, recent, config.precision_alpha)
            system = config.lambda_ * laplacian
            system = system.copy()
            system.flat[:: len(nodes) + 1] += d
            solution = solve_spd_dense(system, d * h)
            left = float(solution[index[edge.a]])
            right = float(solution[index[edge.b]])
            prefix = f"SD_{config.key}"
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
        self.counts["linear_systems"] += len(self.configs)
        return outputs

    def process_row(self, row: pd.Series) -> dict[str, float]:
        now = float(row["TransactionDT"])
        self.release.release_before(now)
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        per_layer = {edge.layer: self._solve_edge(edge, now) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({
                f"{safe}__d{self.delay_days}_ha{self.history_alpha}__{key}": value
                for key, value in values.items()
            })
        for config in self.configs:
            left = [values[f"SD_{config.key}_left"] for values in per_layer.values()]
            right = [values[f"SD_{config.key}_right"] for values in per_layer.values()]
            absdiff = [values[f"SD_{config.key}_absdiff"] for values in per_layer.values()]
            if left:
                endpoints = left + right
                prefix = f"agg__d{self.delay_days}_ha{self.history_alpha}__SD_{config.key}"
                output.update({
                    f"{prefix}_all_sum": float(np.sum(endpoints)),
                    f"{prefix}_all_max": float(np.max(endpoints)),
                    f"{prefix}_all_mean": float(np.mean(endpoints)),
                    f"{prefix}_all_absdiff_max": float(np.max(absdiff)),
                    f"{prefix}_all_absdiff_mean": float(np.mean(absdiff)),
                })
        for edge in edges:
            self.graphs[edge.layer].add_edge(edge.a, edge.b, edge.w)
        self.release.schedule(now, int(row["isFraud"]), edges)
        return output

    def _score_row_from_frozen_state(self, row: pd.Series) -> tuple[dict[str, float], list]:
        now = float(row["TransactionDT"])
        edges = [edge for edge in build_transaction_edges(row) if edge.layer in RISK_LAYERS]
        per_layer = {edge.layer: self._solve_edge(edge, now) for edge in edges}
        output: dict[str, float] = {}
        for layer, values in per_layer.items():
            safe = layer.replace("--", "__")
            output.update({
                f"{safe}__d{self.delay_days}_ha{self.history_alpha}__{key}": value
                for key, value in values.items()
            })
        for config in self.configs:
            left = [values[f"SD_{config.key}_left"] for values in per_layer.values()]
            right = [values[f"SD_{config.key}_right"] for values in per_layer.values()]
            absdiff = [values[f"SD_{config.key}_absdiff"] for values in per_layer.values()]
            if left:
                endpoints = left + right
                prefix = f"agg__d{self.delay_days}_ha{self.history_alpha}__SD_{config.key}"
                output.update({
                    f"{prefix}_all_sum": float(np.sum(endpoints)),
                    f"{prefix}_all_max": float(np.max(endpoints)),
                    f"{prefix}_all_mean": float(np.mean(endpoints)),
                    f"{prefix}_all_absdiff_max": float(np.max(absdiff)),
                    f"{prefix}_all_absdiff_mean": float(np.mean(absdiff)),
                })
        return output, edges

    def process_timestamp_block(self, block: pd.DataFrame) -> list[dict[str, float]]:
        if block.empty:
            return []
        now = float(block.iloc[0]["TransactionDT"])
        self.release.release_before(now)
        scored: list[dict[str, float]] = []
        pending: list[tuple[pd.Series, list]] = []
        for _, row in block.iterrows():
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
        columns = self.expected_columns()
        pending_records: list[dict[str, float]] = []

        def write_records(records: list[dict[str, float]]) -> None:
            nonlocal writer
            if not records:
                return
            batch = pd.DataFrame(records)
            batch = batch.reindex(columns=columns).astype("float32")
            table = pa.Table.from_pandas(batch, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata.update({
                b"causal_policy": b"strict_timestamp_block",
                b"same_timestamp_policy": b"block_frozen_t_minus",
                b"label_release_policy": b"release_time_strictly_less_than_candidate_timestamp",
                b"delay_days": str(self.delay_days).encode(),
                b"history_alpha": str(self.history_alpha).encode(),
                b"ego_radius": str(self.radius).encode(),
                b"max_ego_nodes": str(self.cap).encode(),
                b"cache_version": b"block_causal_v1",
            })
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)

        try:
            for _, block in df.groupby("TransactionDT", sort=False):
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
            "linear_systems": int(self.counts["linear_systems"]),
            "mean_ego_size": float(np.mean(self.ego_sizes)) if self.ego_sizes else 0.0,
            "max_ego_size": int(max(self.ego_sizes)) if self.ego_sizes else 0,
        }


def load_reference_sd(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not path.exists():
        return pd.DataFrame(), {}
    frame = pd.read_parquet(path)
    groups = {}
    for rule in ("count", "sqrt"):
        cols = [c for c in frame.columns if f"SD_{rule}" in c]
        if cols:
            groups[f"exact_{rule}_pa5_l1p0"] = cols
    return frame, groups


def add_recent_shrinkage(green: pd.DataFrame, delays: tuple[int, ...], alpha: int) -> tuple[pd.DataFrame, list[str]]:
    # Cheap uncertainty-gated variants used as a compute-reduction screen. These
    # do not replace exact S_D fields; they let validation detect whether a rule
    # is worth exact solves in a later run.
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
            for rule, precision_left, precision_right in (
                ("log1p_a5", 5.0 + np.log1p(n_left), 5.0 + np.log1p(n_right)),
                ("count_a20", 20.0 + n_left, 20.0 + n_right),
                ("sqrt_a20", 20.0 + np.sqrt(n_left), 20.0 + np.sqrt(n_right)),
                ("count_a100", 100.0 + n_left, 100.0 + n_right),
                ("sqrt_a100", 100.0 + np.sqrt(n_left), 100.0 + np.sqrt(n_right)),
            ):
                gamma_left = 1.0 / precision_left
                gamma_right = 1.0 / precision_right
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


def tuned_two_stage_regions(
    frame: pd.DataFrame,
    y: np.ndarray,
    valid: slice,
    test: slice,
    valid_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    model_names: list[str],
    summary_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_valid = y[valid]
    split = len(y_valid) // 2
    logits_valid = [logit(np.clip(valid_predictions[name], 1e-6, 1 - 1e-6)) for name in model_names]
    logits_test = [logit(np.clip(test_predictions[name], 1e-6, 1 - 1e-6)) for name in model_names]
    raw_valid = frame.iloc[valid][summary_cols].replace([np.inf, -np.inf], np.nan)
    raw_test = frame.iloc[test][summary_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    meta_valid = np.column_stack([*logits_valid, imputer.fit_transform(raw_valid)])
    meta_test = np.column_stack([*logits_test, imputer.transform(raw_test)])
    best = None
    trials = []
    for fraction in (0.025, 0.05, 0.10, 0.20, 0.30):
        first_base = valid_predictions["M3"][:split]
        train_mask = first_base >= np.quantile(first_base, 1 - fraction)
        if np.unique(y_valid[:split][train_mask]).size < 2:
            continue
        model = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0)
        model.fit(meta_valid[:split][train_mask], y_valid[:split][train_mask])
        second_base = valid_predictions["M3"][split:]
        second_mask = second_base >= np.quantile(second_base, 1 - fraction)
        probability = model.predict_proba(meta_valid[split:])[:, 1]
        score = rerank(second_base, probability, second_mask)
        metrics = evaluate(y_valid[split:], score)
        key = (metrics["precision_at_0.01"], metrics["precision_at_0.005"], metrics["auc_pr"])
        trials.append({"fraction": fraction, "selection_key": list(key), "validation_half2": metrics})
        if best is None or key > best[0]:
            best = (key, fraction)
    if best is None:
        return valid_predictions["M3"], test_predictions["M3"], {
            "fallback": "M3",
            "reason": "insufficient positives in validation tail candidates",
            "test_used_for_selection": False,
        }
    fraction = best[1]
    train_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    final = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0)
    final.fit(meta_valid[train_mask], y_valid[train_mask])
    valid_probability = final.predict_proba(meta_valid)[:, 1]
    test_probability = final.predict_proba(meta_test)[:, 1]
    valid_mask = valid_predictions["M3"] >= np.quantile(valid_predictions["M3"], 1 - fraction)
    test_mask = test_predictions["M3"] >= np.quantile(test_predictions["M3"], 1 - fraction)
    valid_score = rerank(valid_predictions["M3"], valid_probability, valid_mask)
    test_score = rerank(test_predictions["M3"], test_probability, test_mask)
    return valid_score, test_score, {
        "selection": "validation split-half P@1%, then P@0.5%, then AUC-PR",
        "candidate_fraction": fraction,
        "selection_key": list(best[0]),
        "trials": trials,
        "validation": evaluate(y_valid, valid_score),
        "test_used_for_selection": False,
    }


class StageTimer:
    def __init__(self) -> None:
        self.started = time.time()
        self.last = self.started
        self.stages: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.time()
        self.stages[name] = self.stages.get(name, 0.0) + now - self.last
        self.last = now

    def total(self) -> float:
        return time.time() - self.started


def stable_model_key(name: str, columns: list[str], seed: int) -> str:
    payload = json.dumps({"name": name, "columns": columns, "seed": seed, "version": 1}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def parse_names(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def screen_model_columns(
    frame: pd.DataFrame,
    y: np.ndarray,
    train: slice,
    valid: slice,
    model_columns: dict[str, list[str]],
    always_fit: set[str],
    keep_top: int,
    seed: int,
    out: Path,
    max_train_rows: int,
    max_iter: int,
    force_screening: bool,
) -> tuple[dict[str, list[str]], dict]:
    if keep_top <= 0 or keep_top >= len(model_columns):
        return model_columns, {
            "enabled": False,
            "reason": "screening disabled or keep_top covers all candidates",
            "selected_models": list(model_columns),
        }
    signature_payload = {
        "models": {name: columns for name, columns in model_columns.items()},
        "always_fit": sorted(always_fit),
        "keep_top": keep_top,
        "max_train_rows": max_train_rows,
        "max_iter": max_iter,
        "seed": seed,
        "version": 2,
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
    screen_path = out / "screening.json"
    if screen_path.exists() and not force_screening:
        try:
            cached = json.loads(screen_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                selected_columns = {
                    name: columns
                    for name, columns in model_columns.items()
                    if name in set(cached.get("selected_models", []))
                }
                cached["cached"] = True
                return selected_columns, cached
        except json.JSONDecodeError:
            pass

    y_valid = y[valid]
    train_indices = np.arange(train.start or 0, train.stop)
    if len(train_indices) > max_train_rows:
        train_indices = np.linspace(train_indices[0], train_indices[-1], max_train_rows, dtype=int)
    candidates = []
    trials = []
    for name, columns in model_columns.items():
        if name == "M3":
            continue
        try:
            preprocessor = make_preprocessor(frame.iloc[train_indices][columns])
            x_train = preprocessor.fit_transform(frame.iloc[train_indices][columns])
            x_valid = preprocessor.transform(frame.iloc[valid][columns])
            model = SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=1e-4,
                max_iter=max_iter,
                tol=1e-3,
                random_state=seed,
            )
            model.fit(x_train, y[train_indices])
            prediction = model.predict_proba(x_valid)[:, 1]
            metrics = evaluate(y_valid, prediction)
            key = (
                metrics["precision_at_0.01"],
                metrics["precision_at_0.005"],
                metrics["auc_pr"],
            )
            trial = {"model": name, "screen_key": list(key), "validation": metrics, "failed": False}
        except Exception as exc:  # screening should not kill the real run
            key = (-1.0, -1.0, -1.0)
            trial = {"model": name, "screen_key": list(key), "failed": True, "error": repr(exc)}
        candidates.append((key, name))
        trials.append(trial)

    selected = {"M3"} | (always_fit & set(model_columns))
    selected.update(name for _, name in sorted(candidates, reverse=True)[:keep_top])
    selected_columns = {name: columns for name, columns in model_columns.items() if name in selected}
    report = {
        "enabled": True,
        "cached": False,
        "signature": signature,
        "keep_top": keep_top,
        "max_train_rows": max_train_rows,
        "max_iter": max_iter,
        "always_fit": sorted(always_fit),
        "selected_models": list(selected_columns),
        "dropped_models": [name for name in model_columns if name not in selected_columns],
        "trials": sorted(trials, key=lambda row: row["screen_key"], reverse=True),
        "test_used_for_selection": False,
    }
    save_json(screen_path, report)
    return selected_columns, report


def fit_named_models_cached(
    frame: pd.DataFrame,
    y: np.ndarray,
    train: slice,
    valid: slice,
    test: slice,
    model_columns: dict[str, list[str]],
    seed: int,
    cache_dir: Path,
    force_models: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    valid_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    selection: dict = {}
    timing: dict[str, dict] = {}
    for name, columns in model_columns.items():
        key = stable_model_key(name, columns, seed)
        npz_path = cache_dir / f"{name}.{key}.npz"
        json_path = cache_dir / f"{name}.{key}.json"
        started = time.time()
        if npz_path.exists() and json_path.exists() and not force_models:
            arrays = np.load(npz_path)
            valid_predictions[name] = arrays["valid"]
            test_predictions[name] = arrays["test"]
            selection[name] = json.loads(json_path.read_text(encoding="utf-8"))
            timing[name] = {"seconds": time.time() - started, "cached": True}
            continue
        valid_score, test_score, details, _ = fit_tail_selected(
            frame.iloc[train][columns],
            y[train],
            frame.iloc[valid][columns],
            y[valid],
            frame.iloc[test][columns],
            seed,
        )
        valid_predictions[name] = valid_score
        test_predictions[name] = test_score
        selection[name] = details
        np.savez_compressed(npz_path, valid=valid_score, test=test_score)
        save_json(json_path, details)
        timing[name] = {"seconds": time.time() - started, "cached": False, "columns": len(columns)}
    return valid_predictions, test_predictions, selection, timing


def run_window(args, window: int) -> dict:
    timer = StageTimer()
    root = Path(args.out_dir)
    out = root / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    baseline_window = Path(args.baseline_dir) / f"window_{window}"
    reference_window = Path(args.reference_dir) / f"window_{window}"
    if args.use_data_cache:
        data = load_ieee_cis_cached(
            args.data_dir,
            (window + 1) * args.window_size,
            cache_dir=args.data_cache_dir,
            force_cache=args.force_data_cache,
        )
    else:
        data = load_ieee_cis(args.data_dir, (window + 1) * args.window_size)
    data = data.iloc[window * args.window_size:(window + 1) * args.window_size].reset_index(drop=True)
    y = data["isFraud"].to_numpy(int)
    train, valid, test = chronological_split(len(data))
    timer.mark("load_data")
    base = pd.read_parquet(baseline_window / "base_features.parquet").reset_index(drop=True)
    schema = pq.ParquetFile(baseline_window / "green_features.parquet").schema.names
    delays = parse_ints(args.delays)
    alpha = args.alpha
    hcols = [c for delay in delays for c in green_columns(schema, delay, alpha, "H")]
    scols = [c for delay in delays for c in green_columns(schema, delay, alpha, "S")]
    green = pd.read_parquet(baseline_window / "green_features.parquet", columns=hcols + scols)
    timer.mark("read_cached_base_green")
    green, sg_cols = add_adaptive_shrinkage(green, delays, alpha)
    green, cheap_cols = add_recent_shrinkage(green, delays, alpha)
    reference_sd, reference_groups = load_reference_sd(reference_window / "precision_weighted_green.parquet")
    timer.mark("derive_cheap_features")
    exact_configs = parse_configs(args.exact_configs)
    exact_path = out / "exact_adaptive_panel.parquet"
    exact_runtime = {"skipped": True, "configs": [config.key for config in exact_configs]}
    if exact_configs:
        if exact_path.exists() and not args.force_features:
            exact_runtime = {"cached": True, "configs": [config.key for config in exact_configs]}
        else:
            builder = ExactAdaptivePanelBuilder(exact_configs, delay_days=0, history_alpha=alpha, recent_days=args.recent_days)
            exact_runtime = builder.write_parquet(data, exact_path)
            exact_runtime["cached"] = False
            exact_runtime["configs"] = [config.key for config in exact_configs]
    exact = pd.read_parquet(exact_path) if exact_path.exists() else pd.DataFrame(index=base.index)
    timer.mark("exact_features")
    frame = pd.concat([base, green, reference_sd, exact], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    baseline_cols = base_groups(list(base.columns))["BP"]
    model_cols: dict[str, list[str]] = {
        "M3": baseline_cols,
        "H": baseline_cols + hcols,
        "H_S": baseline_cols + hcols + scols,
        "adaptive_shrinkage": baseline_cols + hcols + sg_cols,
        "cheap_precision_screen": baseline_cols + hcols + cheap_cols,
    }
    for name, cols in reference_groups.items():
        model_cols[name] = baseline_cols + hcols + cols
    exact_groups: dict[str, list[str]] = {}
    agg_groups: dict[str, list[str]] = {}
    for config in exact_configs:
        key = f"exact_{config.key}"
        cols = [c for c in exact.columns if f"SD_{config.key}" in c]
        if cols:
            exact_groups[key] = cols
            agg_groups[f"{key}_agg"] = [c for c in cols if c.startswith("agg__")]
    for name, cols in exact_groups.items():
        model_cols[name] = baseline_cols + hcols + cols
    for name, cols in agg_groups.items():
        if cols:
            model_cols[name] = baseline_cols + hcols + cols
    timer.mark("assemble_model_columns")
    selected_model_cols, screen_report = screen_model_columns(
        frame,
        y,
        train,
        valid,
        model_cols,
        always_fit=parse_names(args.always_fit_models),
        keep_top=args.screen_top_k,
        seed=args.seed,
        out=out,
        max_train_rows=args.screen_max_train_rows,
        max_iter=args.screen_max_iter,
        force_screening=args.force_screening,
    )
    timer.mark("screen_models")
    valid_predictions, test_predictions, selection, model_fit_timing = fit_named_models_cached(
        frame,
        y,
        train,
        valid,
        test,
        selected_model_cols,
        args.seed,
        cache_dir=out / "model_prediction_cache",
        force_models=args.force_models,
    )
    selection["_screening"] = screen_report
    timer.mark("fit_or_load_models")
    candidate_models = [name for name in valid_predictions if name != "M3"]
    top_for_mix = sorted(candidate_models, key=lambda name: selection_key(y[valid], valid_predictions[name]), reverse=True)[: args.max_mix_components]
    mix_components = ["M3"] + top_for_mix
    mix_valid, mix_test, mix_selection = tuned_soft_mixture(y[valid], valid_predictions, test_predictions, mix_components)
    valid_predictions["adaptive_soft_next"] = mix_valid
    test_predictions["adaptive_soft_next"] = mix_test
    selection["adaptive_soft_next"] = mix_selection
    summary_cols = [
        c for c in sg_cols + cheap_cols + list(reference_sd.columns) + list(exact.columns)
        if c.startswith("agg__") or "all_" in c
    ]
    two_components = ["M3"] + top_for_mix + ["adaptive_soft_next"]
    two_valid, two_test, two_selection = tuned_two_stage_regions(
        frame, y, valid, test, valid_predictions, test_predictions, two_components, summary_cols
    )
    valid_predictions["adaptive_two_stage_next"] = two_valid
    test_predictions["adaptive_two_stage_next"] = two_test
    selection["adaptive_two_stage_next"] = two_selection
    timer.mark("mixture_and_two_stage")
    metrics = {
        name: {"validation": evaluate(y[valid], valid_predictions[name]), "test": evaluate(y[test], test_predictions[name])}
        for name in valid_predictions
    }
    cohorts = cohort_metrics(base.iloc[test].reset_index(drop=True), y[test], test_predictions)
    timer.mark("evaluate_and_cohorts")
    save_json(out / "metrics.json", metrics)
    save_json(out / "selection.json", selection)
    save_json(out / "cohort_metrics.json", {"rows": cohorts})
    runtime = {
        "seconds": timer.total(),
        "window": window,
        "stage_seconds": timer.stages,
        "exact_feature_runtime": exact_runtime,
        "model_fit_timing": model_fit_timing,
        "uses_cached_dense_exact_green": True,
        "uses_reference_sd_cache": bool(reference_groups),
        "uses_data_cache": bool(args.use_data_cache),
        "no_future_labels": True,
        "test_used_for_selection": False,
        "random_forest": "not used",
        "compute_reduction": {
            "reference_count_sqrt_alpha5_reused": bool(reference_groups),
            "multiscale_limited_to_requested_exact_configs": True,
            "lightgbm_candidates_screened_by_validation_logistic": screen_report.get("enabled", False),
            "selected_lightgbm_models": screen_report.get("selected_models", list(selected_model_cols)),
            "dropped_lightgbm_models": screen_report.get("dropped_models", []),
            "model_prediction_cache": True,
            "soft_mixture_components_screened_by_validation": True,
        },
    }
    save_json(out / "runtime.json", runtime)
    rows = []
    for name, values in metrics.items():
        test_metrics = values["test"]
        rows.append({
            "window": window,
            "model": name,
            "auc_pr": test_metrics["auc_pr"],
            "precision_at_0.005": test_metrics["precision_at_0.005"],
            "precision_at_0.01": test_metrics["precision_at_0.01"],
            "precision_at_0.02": test_metrics["precision_at_0.02"],
            "precision_at_0.05": test_metrics["precision_at_0.05"],
            "runtime_seconds": runtime["seconds"],
            "exact_feature_seconds": exact_runtime.get("feature_seconds"),
        })
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({row["model"]: row for row in rows}, indent=2))
    return {"summary": rows, "cohorts": cohorts, "runtime": runtime}


def write_aggregate_outputs(root: Path, reference_dir: Path) -> None:
    summaries = []
    cohorts = []
    runtimes = {}
    for path in sorted(root.glob("window_*/summary.csv")):
        summaries.extend(pd.read_csv(path).to_dict("records"))
    for path in sorted(root.glob("window_*/cohort_metrics.json")):
        window = int(path.parent.name.split("_")[-1])
        rows = json.loads(path.read_text())["rows"]
        for row in rows:
            row["window"] = window
        cohorts.extend(rows)
    for path in sorted(root.glob("window_*/runtime.json")):
        runtimes[path.parent.name] = json.loads(path.read_text())
    summary = pd.DataFrame(summaries).sort_values(["window", "model"])
    cohort_frame = pd.DataFrame(cohorts).sort_values(["window", "cohort", "model"])
    summary.to_csv(root / "window_summary.csv", index=False)
    cohort_frame.to_csv(root / "cohort_metrics.csv", index=False)
    save_json(root / "runtime.json", runtimes)
    reference = pd.read_csv(reference_dir / "window_summary.csv")
    reference = reference[reference["model"] == "adaptive_two_stage"].set_index("window")
    baseline = summary[summary["model"] == "M3"].set_index("window")
    rows = []
    for model, group in summary.groupby("model"):
        if model == "M3":
            continue
        group = group.set_index("window").sort_index()
        common = group.index.intersection(baseline.index)
        ref_common = group.index.intersection(reference.index)
        out = {"model": model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            gain = group.loc[common, metric] - baseline.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(gain.mean())
            out[f"win_count_{metric}"] = int((gain > 0).sum())
            ref_delta = group.loc[ref_common, metric] - reference.loc[ref_common, metric]
            out[f"mean_delta_ref_{metric}"] = float(ref_delta.mean())
            out[f"win_count_ref_{metric}"] = int((ref_delta > 0).sum())
        out["accepted_vs_ref_p01"] = bool(
            (out["mean_delta_ref_precision_at_0.01"] > 0 or out["mean_delta_ref_auc_pr"] > 0)
            and out["win_count_ref_precision_at_0.01"] >= 4
        )
        rows.append(out)
    pd.DataFrame(rows).sort_values("mean_delta_ref_precision_at_0.01", ascending=False).to_csv(root / "mean_gains_vs_reference.csv", index=False)
    critical = cohort_frame[cohort_frame["cohort"].isin(["known_endpoints", "C00_newedge"])]
    critical_rows = []
    ref_cohort_path = reference_dir / "cohort_metrics.csv"
    ref_cohorts = pd.read_csv(ref_cohort_path)
    ref_cohorts = ref_cohorts[ref_cohorts["model"] == "adaptive_two_stage"]
    for (cohort, model), group in critical.groupby(["cohort", "model"]):
        if model == "M3":
            continue
        base_group = critical[(critical["cohort"] == cohort) & (critical["model"] == "M3")].set_index("window")
        ref_group = ref_cohorts[ref_cohorts["cohort"] == cohort].set_index("window")
        group = group.set_index("window").sort_index()
        common = group.index.intersection(base_group.index)
        ref_common = group.index.intersection(ref_group.index)
        out = {"cohort": cohort, "model": model, "windows": int(len(common))}
        for metric in ["auc_pr", "precision_at_0.005", "precision_at_0.01", "precision_at_0.02", "precision_at_0.05"]:
            gain = group.loc[common, metric] - base_group.loc[common, metric]
            out[f"mean_gain_{metric}"] = float(gain.mean())
            out[f"win_count_{metric}"] = int((gain > 0).sum())
            ref_delta = group.loc[ref_common, metric] - ref_group.loc[ref_common, metric]
            out[f"mean_delta_ref_{metric}"] = float(ref_delta.mean())
            out[f"win_count_ref_{metric}"] = int((ref_delta > 0).sum())
        critical_rows.append(out)
    pd.DataFrame(critical_rows).sort_values(
        ["cohort", "mean_delta_ref_precision_at_0.01"], ascending=[True, False]
    ).to_csv(root / "critical_cohort_vs_reference.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--baseline-dir", default="outputs/ieee_green_moderate_100k_v1")
    parser.add_argument("--reference-dir", default="outputs/ieee_green_adaptive_theory_v1")
    parser.add_argument("--out-dir", default="outputs/ieee_green_adaptive_next_round_v1")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--no-data-cache", dest="use_data_cache", action="store_false")
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument(
        "--exact-configs",
        default="log1p:5:1.0,recent:5:1.0,sqrt_recent:5:1.0,sqrt:5:0.03,sqrt:5:0.1,sqrt:5:0.3",
        help="comma-separated rule:precision_alpha:lambda entries; count/sqrt alpha5 lambda1 are reused from reference",
    )
    parser.add_argument("--max-mix-components", type=int, default=5)
    parser.add_argument(
        "--screen-top-k",
        type=int,
        default=8,
        help="validation-logistic screen keeps this many non-always-fit candidates for LightGBM; use 0 to disable screening",
    )
    parser.add_argument("--screen-max-train-rows", type=int, default=20000)
    parser.add_argument("--screen-max-iter", type=int, default=40)
    parser.add_argument("--force-screening", action="store_true")
    parser.add_argument(
        "--always-fit-models",
        default="M3,H,H_S,adaptive_shrinkage,exact_count_pa5_l1p0,exact_sqrt_pa5_l1p0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    parser.set_defaults(use_data_cache=True)
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_existing:
        write_aggregate_outputs(root, Path(args.reference_dir))
        return
    for window in parse_ints(args.windows):
        run_window(args, window)
    write_aggregate_outputs(root, Path(args.reference_dir))


if __name__ == "__main__":
    main()

