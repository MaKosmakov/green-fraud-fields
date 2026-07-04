from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(len(y) * frac)))
    return float(np.mean(y[np.argsort(-score)[:k]]))


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auc_pr": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "p_at_0.005": precision_at(y, score, 0.005),
        "p_at_0.01": precision_at(y, score, 0.01),
        "p_at_0.02": precision_at(y, score, 0.02),
        "p_at_0.05": precision_at(y, score, 0.05),
        "p_at_100": float(np.mean(y[np.argsort(-score)[: min(100, len(y))]])),
        "p_at_500": float(np.mean(y[np.argsort(-score)[: min(500, len(y))]])),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int,
    fast: bool,
) -> LGBMClassifier:
    pos = max(float(np.sum(y_train)), 1.0)
    scale = (len(y_train) - pos) / pos
    model = LGBMClassifier(
        n_estimators=300 if fast else 700,
        learning_rate=0.05 if fast else 0.03,
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
        callbacks=[early_stopping(40 if fast else 60, verbose=False)],
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


def local_nodes(
    seeds: np.ndarray,
    adj: dict[int, set[int]],
    n: np.ndarray,
    radius: int,
    cap: int,
) -> list[int]:
    if len(seeds) == 0:
        return []
    seed_list = [int(x) for x in seeds]
    seen = set(seed_list)
    order = list(seed_list)
    q: deque[tuple[int, int]] = deque((x, 0) for x in seed_list)
    max_collect = max(cap * 3, len(seed_list))
    while q:
        node, dist = q.popleft()
        if dist >= radius:
            continue
        neigh = adj.get(node)
        if not neigh:
            continue
        for nb in neigh:
            nb = int(nb)
            if nb not in seen:
                seen.add(nb)
                order.append(nb)
                if len(order) >= max_collect:
                    q.clear()
                    break
                q.append((nb, dist + 1))
    if len(order) <= cap:
        return order
    seed_set = set(seed_list)
    non_seed = [x for x in order if x not in seed_set]
    non_seed.sort(key=lambda x: (-n[x], x))
    return seed_list[:cap] + non_seed[: max(0, cap - len(seed_list))]


def exact_green_values(
    addrs: np.ndarray | None,
    adj: dict[int, set[int]],
    n: np.ndarray,
    f: np.ndarray,
    p0: float,
    alpha: float,
    radius: int,
    cap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if addrs is None or len(addrs) == 0:
        h = np.array([p0], dtype=np.float64)
        return h, h.copy(), 0
    addrs = np.asarray(addrs, dtype=np.int64)
    h_endpoint = (f[addrs] + alpha * p0) / (n[addrs] + alpha)
    nodes = local_nodes(addrs, adj, n, radius=radius, cap=cap)
    if not nodes:
        return h_endpoint, h_endpoint.copy(), 0
    idx = {node: i for i, node in enumerate(nodes)}
    node_arr = np.asarray(nodes, dtype=np.int64)
    h = (f[node_arr] + alpha * p0) / (n[node_arr] + alpha)
    d = alpha + np.sqrt(n[node_arr])
    m = len(nodes)
    a = np.diag(d.copy())
    for u in nodes:
        i = idx[u]
        for v in adj.get(u, ()):
            j = idx.get(int(v))
            if j is None or j == i:
                continue
            a[i, i] += 1.0
            a[i, j] -= 1.0
    rhs = d * h
    try:
        s = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(a, rhs, rcond=None)[0]
    s_endpoint = np.array([s[idx[int(x)]] if int(x) in idx else (f[int(x)] + alpha * p0) / (n[int(x)] + alpha) for x in addrs])
    return h_endpoint, s_endpoint, m


def build_exact_green_features(
    frame: pd.DataFrame,
    input_map: dict[int, np.ndarray],
    output_map: dict[int, np.ndarray],
    all_map: dict[int, np.ndarray],
    n_addr: int,
    alpha: float,
    prior: float,
    max_bipartite_edges: int,
    radius: int,
    cap: int,
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
    started = time.time()
    for step in sorted(tx_ids_by_time):
        p0 = released_pos / released_total if released_total > 0 else prior
        block_rows: list[dict] = []
        solve_nodes: list[int] = []
        for tx in tx_ids_by_time[step]:
            addrs = all_map.get(tx)
            h_vals, s_vals, m = exact_green_values(addrs, adj, n, f, p0, alpha, radius, cap)
            n_vals = np.array([0.0]) if addrs is None or len(addrs) == 0 else n[addrs]
            solve_nodes.append(m)
            block_rows.append(
                {
                    "txId": tx,
                    "ell_h_mean": float(np.mean(h_vals)),
                    "ell_h_max": float(np.max(h_vals)),
                    "ell_h_min": float(np.min(h_vals)),
                    "ell_h_std": float(np.std(h_vals)),
                    "ell_sd_mean": float(np.mean(s_vals)),
                    "ell_sd_max": float(np.max(s_vals)),
                    "ell_sd_min": float(np.min(s_vals)),
                    "ell_sd_std": float(np.std(s_vals)),
                    "ell_sd_minus_h_mean": float(np.mean(s_vals) - np.mean(h_vals)),
                    "ell_sd_absdiff_mean": float(np.mean(np.abs(s_vals - h_vals))),
                    "ell_addr_n_sum": float(np.sum(n_vals)),
                    "ell_addr_n_max": float(np.max(n_vals)),
                    "ell_endpoint_count": int(0 if addrs is None else len(addrs)),
                    "ell_prior_rate": float(p0),
                    "ell_green_nodes": int(m),
                }
            )

        # Strict block-causal update after scoring all rows in this time step.
        for tx in tx_ids_by_time[step]:
            ins = input_map.get(tx)
            outs = output_map.get(tx)
            if ins is not None and outs is not None and len(ins) and len(outs):
                product = len(ins) * len(outs)
                if product <= max_bipartite_edges:
                    for aa in ins:
                        a_int = int(aa)
                        for bb in outs:
                            b_int = int(bb)
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
                    "elapsed_sec": round(time.time() - started, 1),
                    "block_txs": len(tx_ids_by_time[step]),
                    "released_total_after": int(released_total),
                    "mean_solve_nodes": round(float(np.mean(solve_nodes)), 2) if solve_nodes else 0,
                    "p95_solve_nodes": round(float(np.percentile(solve_nodes, 95)), 2) if solve_nodes else 0,
                    "graph_nodes_with_edges": len(adj),
                }
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/elliptic_plus_plus/google_drive")
    parser.add_argument("--out-dir", default="outputs/elliptic_plus_plus_exact_green_v1")
    parser.add_argument("--train-end", type=int, default=30)
    parser.add_argument("--valid-end", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--prior", type=float, default=0.05)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--cap", type=int, default=50)
    parser.add_argument("--max-bipartite-edges", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast-model", action="store_true")
    args = parser.parse_args()

    started = time.time()
    raw = Path(args.raw)
    tx_dir = raw / "Transactions Dataset"
    out = Path(args.out_dir) / f"r{args.radius}_cap{args.cap}_alpha{args.alpha:g}"
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

    feature_path = out / "exact_green_features.parquet"
    if feature_path.exists():
        print(f"using cached {feature_path}", flush=True)
        green = pd.read_parquet(feature_path)
    else:
        print("building exact local Green features", flush=True)
        green = build_exact_green_features(
            frame[["txId", "Time step", "known", "y"]],
            input_map,
            output_map,
            all_map,
            n_addr,
            alpha=args.alpha,
            prior=args.prior,
            max_bipartite_edges=args.max_bipartite_edges,
            radius=args.radius,
            cap=args.cap,
        )
        green.to_parquet(feature_path, index=False)
    frame = frame.merge(green, on="txId", how="left")

    metadata_cols = {"txId", "class", "known", "y", "split"}
    aggregate_cols = {c for c in frame.columns if c.startswith("Aggregate_feature_")}
    h_cols = [
        "ell_h_mean",
        "ell_h_max",
        "ell_h_min",
        "ell_h_std",
        "ell_addr_n_sum",
        "ell_addr_n_max",
        "ell_endpoint_count",
        "ell_prior_rate",
    ]
    sd_cols = [
        "ell_sd_mean",
        "ell_sd_max",
        "ell_sd_min",
        "ell_sd_std",
        "ell_sd_minus_h_mean",
        "ell_sd_absdiff_mean",
        "ell_green_nodes",
    ]
    base_cols = [
        c
        for c in frame.columns
        if c not in metadata_cols and c not in aggregate_cols and not c.startswith("ell_")
    ]

    known = frame[frame["known"]].copy()
    train = known[known["split"] == "train"]
    valid = known[known["split"] == "valid"]
    test = known[known["split"] == "test"]
    y_train = train["y"].to_numpy()
    y_valid = valid["y"].to_numpy()
    y_test = test["y"].to_numpy()

    model_specs = {
        "strict_features": base_cols,
        "strict_features_H_raw": base_cols + h_cols,
        "strict_features_H_raw_S_D_exact": base_cols + h_cols + sd_cols,
    }
    rows = []
    predictions = test[["txId", "Time step", "class", "y"]].copy()
    for name, cols in model_specs.items():
        print(f"fitting {name} with {len(cols)} columns", flush=True)
        model = fit_lgbm(train[cols], y_train, valid[cols], y_valid, args.seed, args.fast_model)
        pred = model.predict_proba(test[cols])[:, 1]
        predictions[f"score_{name}"] = pred
        row = metrics(y_test, pred)
        row.update({"model": name, "n_features": len(cols)})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    predictions.to_parquet(out / "test_predictions.parquet", index=False)

    baseline = summary.set_index("model").loc["strict_features"]
    history = summary.set_index("model").loc["strict_features_H_raw"]
    gains = []
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
                "accepted_graph_marginal": bool(row["model"] == "strict_features_H_raw_S_D_exact" and row["auc_pr"] > history["auc_pr"] and row["p_at_0.01"] >= history["p_at_0.01"]),
            }
        )
    gains_df = pd.DataFrame(gains)
    gains_df.to_csv(out / "gains.csv", index=False)

    audit = {
        "dataset": "Elliptic++ transactions plus address-transaction graph",
        "causal_policy": "strict_time_step_block",
        "label_policy": "classes 1/2 known labels only; class 3 unknown excluded from train/eval and never used as negative",
        "wallet_label_policy": "official wallets_classes.csv not used as features",
        "feature_policy": "published transaction aggregate features excluded; local and transaction-stat columns retained",
        "green_policy": "exact dense local solve S_D=(L+D)^-1 D H on historical address graph",
        "radius": args.radius,
        "cap": args.cap,
        "alpha": args.alpha,
        "prior": args.prior,
        "splits": {"train_time_steps": f"1-{args.train_end}", "valid_time_steps": f"{args.train_end + 1}-{args.valid_end}", "test_time_steps": f"{args.valid_end + 1}-49"},
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
        "# Elliptic++ exact local Green check",
        "",
        "This is the narrow fair follow-up to the Elliptic++ smoke: raw released address history versus exact local adaptive Green smoothing.",
        "",
        "Leakage controls:",
        "- strict time-step block processing;",
        "- no official wallet labels as features;",
        "- unknown class-3 transactions excluded from metrics and never treated as negatives;",
        "- labels and graph edges from a time step update state only after the full block is scored;",
        "- published aggregate transaction features excluded from the strict feature baseline.",
        "",
        f"Exact Green setting: radius={args.radius}, cap={args.cap}, alpha={args.alpha:g}.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Gains",
        "",
        markdown_table(gains_df),
        "",
        "Acceptance rule: exact `S_D` must beat raw address history, not merely the transaction-feature baseline.",
    ]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(gains_df.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
