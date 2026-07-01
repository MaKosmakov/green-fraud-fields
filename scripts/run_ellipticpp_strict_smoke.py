from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def precision_at(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    if len(y) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y) * frac)))
    idx = np.argsort(-score)[:k]
    return float(np.mean(y[idx]))


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)) if len(y) else float("nan"),
        "auc_pr": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "p_at_0.005": precision_at(y, score, 0.005),
        "p_at_0.01": precision_at(y, score, 0.01),
        "p_at_0.02": precision_at(y, score, 0.02),
        "p_at_0.05": precision_at(y, score, 0.05),
        "p_at_100": float(np.mean(y[np.argsort(-score)[: min(100, len(y))]])) if len(y) else float("nan"),
        "p_at_500": float(np.mean(y[np.argsort(-score)[: min(500, len(y))]])) if len(y) else float("nan"),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def fit_lgbm(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame, y_valid: np.ndarray, seed: int) -> LGBMClassifier:
    pos = max(float(np.sum(y_train)), 1.0)
    scale = (len(y_train) - pos) / pos
    model = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[early_stopping(60, verbose=False)],
    )
    return model


def load_address_maps(raw: Path, tx_ids: set[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], int]:
    actor_dir = raw / "Actors Dataset"
    addr_tx = pd.read_csv(actor_dir / "AddrTx_edgelist.csv")
    tx_addr = pd.read_csv(actor_dir / "TxAddr_edgelist.csv")
    addr_tx = addr_tx[addr_tx["txId"].isin(tx_ids)].copy()
    tx_addr = tx_addr[tx_addr["txId"].isin(tx_ids)].copy()

    all_addr = pd.concat([addr_tx["input_address"], tx_addr["output_address"]], ignore_index=True)
    codes, _ = pd.factorize(all_addr, sort=False)
    addr_tx["addr_id"] = codes[: len(addr_tx)].astype(np.int64)
    tx_addr["addr_id"] = codes[len(addr_tx) :].astype(np.int64)
    n_addr = int(codes.max()) + 1 if len(codes) else 0

    input_map = {
        int(k): np.array(sorted(set(v)), dtype=np.int64)
        for k, v in addr_tx.groupby("txId")["addr_id"].agg(list).items()
    }
    output_map = {
        int(k): np.array(sorted(set(v)), dtype=np.int64)
        for k, v in tx_addr.groupby("txId")["addr_id"].agg(list).items()
    }
    all_map: dict[int, np.ndarray] = {}
    for tx in set(input_map) | set(output_map):
        if tx in input_map and tx in output_map:
            all_map[tx] = np.array(sorted(set(input_map[tx]).union(output_map[tx])), dtype=np.int64)
        elif tx in input_map:
            all_map[tx] = input_map[tx]
        else:
            all_map[tx] = output_map[tx]
    return input_map, output_map, all_map, n_addr


def build_causal_history_features(
    frame: pd.DataFrame,
    input_map: dict[int, np.ndarray],
    output_map: dict[int, np.ndarray],
    all_map: dict[int, np.ndarray],
    n_addr: int,
    alpha: float,
    prior: float,
    max_bipartite_edges: int,
) -> pd.DataFrame:
    n = np.zeros(n_addr, dtype=np.float64)
    f = np.zeros(n_addr, dtype=np.float64)
    adj: dict[int, set[int]] = defaultdict(set)
    tx_ids_by_time = {
        int(t): list(g["txId"].astype(int).to_numpy())
        for t, g in frame.sort_values(["Time step", "txId"]).groupby("Time step")
    }
    y_by_tx = dict(zip(frame["txId"].astype(int), frame["y"].astype(float)))
    known_by_tx = dict(zip(frame["txId"].astype(int), frame["known"].astype(bool)))

    rows: list[dict] = []
    released_pos = 0.0
    released_total = 0.0
    t0 = time.time()
    for step in sorted(tx_ids_by_time):
        p0 = released_pos / released_total if released_total > 0 else prior
        block_rows: list[dict] = []
        for tx in tx_ids_by_time[step]:
            addrs = all_map.get(tx)
            if addrs is None or len(addrs) == 0:
                h_vals = np.array([p0])
                s_vals = h_vals
                n_vals = np.array([0.0])
                endpoint_count = 0
            else:
                h_vals = (f[addrs] + alpha * p0) / (n[addrs] + alpha)
                n_vals = n[addrs]
                s_vals_list = []
                for a, h_a in zip(addrs, h_vals):
                    neigh = adj.get(int(a))
                    if neigh:
                        neigh_arr = np.fromiter(neigh, dtype=np.int64)
                        h_neigh = (f[neigh_arr] + alpha * p0) / (n[neigh_arr] + alpha)
                        d_a = alpha + np.sqrt(n[int(a)])
                        s_a = (d_a * h_a + float(np.sum(h_neigh))) / (d_a + len(neigh_arr))
                    else:
                        s_a = h_a
                    s_vals_list.append(float(s_a))
                s_vals = np.asarray(s_vals_list, dtype=np.float64)
                endpoint_count = int(len(addrs))
            block_rows.append(
                {
                    "txId": tx,
                    "ell_h_mean": float(np.mean(h_vals)),
                    "ell_h_max": float(np.max(h_vals)),
                    "ell_h_min": float(np.min(h_vals)),
                    "ell_h_std": float(np.std(h_vals)),
                    "ell_s_mean": float(np.mean(s_vals)),
                    "ell_s_max": float(np.max(s_vals)),
                    "ell_s_min": float(np.min(s_vals)),
                    "ell_s_std": float(np.std(s_vals)),
                    "ell_s_minus_h_mean": float(np.mean(s_vals) - np.mean(h_vals)),
                    "ell_addr_n_sum": float(np.sum(n_vals)),
                    "ell_addr_n_max": float(np.max(n_vals)),
                    "ell_endpoint_count": endpoint_count,
                    "ell_prior_rate": float(p0),
                }
            )

        # Conservative timestamp-block policy: only after all txs in this time step
        # are scored do graph edges and known labels enter future state.
        for tx in tx_ids_by_time[step]:
            ins = input_map.get(tx)
            outs = output_map.get(tx)
            if ins is not None and outs is not None and len(ins) and len(outs):
                product = len(ins) * len(outs)
                if product <= max_bipartite_edges:
                    for a in ins:
                        a_int = int(a)
                        for b in outs:
                            b_int = int(b)
                            if a_int != b_int:
                                adj[a_int].add(b_int)
                                adj[b_int].add(a_int)
            if known_by_tx.get(tx, False):
                addrs = all_map.get(tx)
                if addrs is not None and len(addrs):
                    y = y_by_tx[tx]
                    n[addrs] += 1.0
                    f[addrs] += y
                released_total += 1.0
                released_pos += y_by_tx[tx]
        rows.extend(block_rows)
        print(
            json.dumps(
                {
                    "step": step,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "block_txs": len(tx_ids_by_time[step]),
                    "released_total_after": int(released_total),
                    "graph_nodes_with_edges": len(adj),
                }
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/elliptic_plus_plus/google_drive")
    parser.add_argument("--out-dir", default="outputs/elliptic_plus_plus_strict_smoke_v1")
    parser.add_argument("--train-end", type=int, default=30)
    parser.add_argument("--valid-end", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--prior", type=float, default=0.05)
    parser.add_argument("--max-bipartite-edges", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    raw = Path(args.raw)
    tx_dir = raw / "Transactions Dataset"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("reading transaction features/classes", flush=True)
    features = pd.read_csv(tx_dir / "txs_features.csv")
    classes = pd.read_csv(tx_dir / "txs_classes.csv")
    frame = features.merge(classes, on="txId", how="inner")
    frame["known"] = frame["class"].isin([1, 2])
    frame["y"] = (frame["class"] == 1).astype(int)
    frame["split"] = np.select(
        [frame["Time step"] <= args.train_end, frame["Time step"] <= args.valid_end],
        ["train", "valid"],
        default="test",
    )

    print("reading address maps", flush=True)
    input_map, output_map, all_map, n_addr = load_address_maps(raw, set(frame["txId"].astype(int)))
    print(json.dumps({"n_transactions": len(frame), "n_addresses_in_tx_maps": n_addr, "mapped_txs": len(all_map)}), flush=True)

    hist_path = out / "causal_history_features.parquet"
    if hist_path.exists():
        print(f"using cached {hist_path}", flush=True)
        hist = pd.read_parquet(hist_path)
    else:
        print("building strict block-causal address history/Green-smoke features", flush=True)
        hist = build_causal_history_features(
            frame[["txId", "Time step", "known", "y"]],
            input_map,
            output_map,
            all_map,
            n_addr,
            alpha=args.alpha,
            prior=args.prior,
            max_bipartite_edges=args.max_bipartite_edges,
        )
        hist.to_parquet(hist_path, index=False)
    frame = frame.merge(hist, on="txId", how="left")

    metadata_cols = {"txId", "class", "known", "y", "split"}
    aggregate_cols = {c for c in frame.columns if c.startswith("Aggregate_feature_")}
    graph_cols_h = ["ell_h_mean", "ell_h_max", "ell_h_min", "ell_h_std", "ell_addr_n_sum", "ell_addr_n_max", "ell_endpoint_count", "ell_prior_rate"]
    graph_cols_s = ["ell_s_mean", "ell_s_max", "ell_s_min", "ell_s_std", "ell_s_minus_h_mean"]
    base_cols = [
        c
        for c in frame.columns
        if c not in metadata_cols
        and c not in aggregate_cols
        and not c.startswith("ell_")
    ]

    # Unknown labels are not negatives and are excluded from train/validation/test metrics.
    known = frame[frame["known"]].copy()
    train = known[known["split"] == "train"]
    valid = known[known["split"] == "valid"]
    test = known[known["split"] == "test"]
    y_train = train["y"].to_numpy()
    y_valid = valid["y"].to_numpy()
    y_test = test["y"].to_numpy()

    model_specs = {
        "strict_features": base_cols,
        "strict_features_H_raw": base_cols + graph_cols_h,
        "strict_features_H_raw_S_neighbor": base_cols + graph_cols_h + graph_cols_s,
    }
    predictions: dict[str, np.ndarray] = {}
    rows = []
    for name, cols in model_specs.items():
        print(f"fitting {name} with {len(cols)} columns", flush=True)
        model = fit_lgbm(train[cols], y_train, valid[cols], y_valid, args.seed)
        pred = model.predict_proba(test[cols])[:, 1]
        predictions[name] = pred
        m = metrics(y_test, pred)
        m.update({"model": name, "n_features": len(cols)})
        rows.append(m)

    print("fitting fixed top-20 validation logistic tail reranker", flush=True)
    base_name = "strict_features_H_raw_S_neighbor"
    tail_cols = ["base_score"] + graph_cols_h + graph_cols_s
    base_valid_model = fit_lgbm(train[model_specs[base_name]], y_train, valid[model_specs[base_name]], y_valid, args.seed)
    valid_score = base_valid_model.predict_proba(valid[model_specs[base_name]])[:, 1]
    test_score = base_valid_model.predict_proba(test[model_specs[base_name]])[:, 1]
    valid_tail_cut = np.quantile(valid_score, 0.80)
    train_tail_frame = valid[graph_cols_h + graph_cols_s].copy()
    train_tail_frame.insert(0, "base_score", valid_score)
    test_tail_frame = test[graph_cols_h + graph_cols_s].copy()
    test_tail_frame.insert(0, "base_score", test_score)
    tail_mask_valid = valid_score >= valid_tail_cut
    tail = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))
    if len(np.unique(y_valid[tail_mask_valid])) > 1:
        tail.fit(train_tail_frame.loc[tail_mask_valid, tail_cols], y_valid[tail_mask_valid])
        tail_test_score = test_score.copy()
        tail_mask_test = test_score >= valid_tail_cut
        if tail_mask_test.any():
            tail_test_score[tail_mask_test] = tail.predict_proba(test_tail_frame.loc[tail_mask_test, tail_cols])[:, 1]
        predictions["fixed_top20_tail"] = tail_test_score
        m = metrics(y_test, tail_test_score)
        m.update({"model": "fixed_top20_tail", "n_features": len(tail_cols), "valid_tail_cut": float(valid_tail_cut), "test_tail_n": int(tail_mask_test.sum())})
        rows.append(m)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pred_frame = test[["txId", "Time step", "class", "y"]].copy()
    for name, pred in predictions.items():
        pred_frame[f"score_{name}"] = pred
    pred_frame.to_parquet(out / "test_predictions.parquet", index=False)

    gains = []
    baseline = summary.set_index("model").loc["strict_features"]
    history = summary.set_index("model").loc["strict_features_H_raw"]
    for _, row in summary.iterrows():
        if row["model"] == "strict_features":
            continue
        gains.append(
            {
                "model": row["model"],
                "auc_pr_gain_vs_features": row["auc_pr"] - baseline["auc_pr"],
                "p_at_0.01_gain_vs_features": row["p_at_0.01"] - baseline["p_at_0.01"],
                "auc_pr_gain_vs_H_raw": row["auc_pr"] - history["auc_pr"],
                "p_at_0.01_gain_vs_H_raw": row["p_at_0.01"] - history["p_at_0.01"],
            }
        )
    pd.DataFrame(gains).to_csv(out / "gains.csv", index=False)

    audit = {
        "dataset": "Elliptic++ transactions plus address-transaction graph",
        "causal_policy": "strict_time_step_block",
        "label_policy": "classes 1/2 known labels only; class 3 unknown excluded from train/eval and never used as negative",
        "wallet_label_policy": "official wallets_classes.csv not downloaded/used as features",
        "feature_policy": "published transaction aggregate features excluded; local and transaction-stat columns retained",
        "green_policy": "fast smoke: one-hop adaptive neighbor/Jacobi field on historical address graph, not final exact radius-2 Green solve",
        "splits": {"train_time_steps": f"1-{args.train_end}", "valid_time_steps": f"{args.train_end + 1}-{args.valid_end}", "test_time_steps": f"{args.valid_end + 1}-49"},
        "alpha": args.alpha,
        "prior": args.prior,
        "runtime_sec": time.time() - started,
        "counts": {
            "all_rows": int(len(frame)),
            "known_rows": int(len(known)),
            "train_known": int(len(train)),
            "valid_known": int(len(valid)),
            "test_known": int(len(test)),
            "train_pos": int(y_train.sum()),
            "valid_pos": int(y_valid.sum()),
            "test_pos": int(y_test.sum()),
        },
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = [
        "# Elliptic++ strict temporal smoke check",
        "",
        "This is a bounded second-dataset viability check, not a final paper-grade exact Green run.",
        "",
        "Leakage controls:",
        "- strict time-step block processing;",
        "- no official wallet labels as features;",
        "- unknown class-3 transactions excluded from metrics and never treated as negatives;",
        "- labels from a time step update address history only after that full block is scored;",
        "- published aggregate transaction features excluded from the baseline feature set.",
        "",
        "Important caveat: `S_neighbor` is a fast one-hop adaptive neighbor/Jacobi field, not the final exact radius-2/cap-100 Green solve.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Gains",
        "",
        markdown_table(pd.DataFrame(gains)),
    ]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

