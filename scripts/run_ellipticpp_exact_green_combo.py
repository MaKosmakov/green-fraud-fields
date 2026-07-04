from __future__ import annotations

import argparse
import json
import time
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
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/elliptic_plus_plus/google_drive")
    parser.add_argument("--exact-root", default="outputs/elliptic_plus_plus_exact_green_v1")
    parser.add_argument("--out-dir", default="outputs/elliptic_plus_plus_exact_green_v1/combo_H5_exact")
    parser.add_argument("--train-end", type=int, default=30)
    parser.add_argument("--valid-end", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    raw = Path(args.raw)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("reading transactions", flush=True)
    tx_dir = raw / "Transactions Dataset"
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

    root = Path(args.exact_root)
    h5 = pd.read_parquet(root / "r1_cap50_alpha5" / "exact_green_features.parquet")
    sd5 = h5[["txId", "ell_sd_mean", "ell_sd_max", "ell_sd_min", "ell_sd_std", "ell_sd_minus_h_mean", "ell_sd_absdiff_mean", "ell_green_nodes"]].copy()
    sd20 = pd.read_parquet(root / "r1_cap50_alpha20" / "exact_green_features.parquet")[
        ["txId", "ell_sd_mean", "ell_sd_max", "ell_sd_min", "ell_sd_std", "ell_sd_minus_h_mean", "ell_sd_absdiff_mean", "ell_green_nodes"]
    ].copy()
    sd2 = pd.read_parquet(root / "r2_cap50_alpha5" / "exact_green_features.parquet")[
        ["txId", "ell_sd_mean", "ell_sd_max", "ell_sd_min", "ell_sd_std", "ell_sd_minus_h_mean", "ell_sd_absdiff_mean", "ell_green_nodes"]
    ].copy()

    h_cols = ["ell_h_mean", "ell_h_max", "ell_h_min", "ell_h_std", "ell_addr_n_sum", "ell_addr_n_max", "ell_endpoint_count", "ell_prior_rate"]
    h5 = h5[["txId"] + h_cols].rename(columns={c: f"h5_{c}" for c in h_cols})
    sd5 = sd5.rename(columns={c: f"sd_r1a5_{c}" for c in sd5.columns if c != "txId"})
    sd20 = sd20.rename(columns={c: f"sd_r1a20_{c}" for c in sd20.columns if c != "txId"})
    sd2 = sd2.rename(columns={c: f"sd_r2a5_{c}" for c in sd2.columns if c != "txId"})

    frame = frame.merge(h5, on="txId", how="left").merge(sd5, on="txId", how="left").merge(sd20, on="txId", how="left").merge(sd2, on="txId", how="left")

    metadata_cols = {"txId", "class", "known", "y", "split"}
    aggregate_cols = {c for c in frame.columns if c.startswith("Aggregate_feature_")}
    base_cols = [c for c in frame.columns if c not in metadata_cols and c not in aggregate_cols and not c.startswith(("h5_", "sd_"))]
    h5_cols = [c for c in frame.columns if c.startswith("h5_")]
    sd5_cols = [c for c in frame.columns if c.startswith("sd_r1a5_")]
    sd20_cols = [c for c in frame.columns if c.startswith("sd_r1a20_")]
    sd2_cols = [c for c in frame.columns if c.startswith("sd_r2a5_")]

    known = frame[frame["known"]].copy()
    train = known[known["split"] == "train"]
    valid = known[known["split"] == "valid"]
    test = known[known["split"] == "test"]
    y_train = train["y"].to_numpy()
    y_valid = valid["y"].to_numpy()
    y_test = test["y"].to_numpy()

    specs = {
        "strict_features": base_cols,
        "H_raw_alpha5": base_cols + h5_cols,
        "H5_plus_SD_r1_alpha5": base_cols + h5_cols + sd5_cols,
        "H5_plus_SD_r2_alpha5": base_cols + h5_cols + sd2_cols,
        "H5_plus_SD_r1_alpha20": base_cols + h5_cols + sd20_cols,
    }
    rows = []
    pred = test[["txId", "Time step", "class", "y"]].copy()
    for name, cols in specs.items():
        print(f"fitting {name} ({len(cols)} cols)", flush=True)
        model = fit_lgbm(train[cols], y_train, valid[cols], y_valid, args.seed)
        score = model.predict_proba(test[cols])[:, 1]
        pred[f"score_{name}"] = score
        row = metrics(y_test, score)
        row.update({"model": name, "n_features": len(cols)})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pred.to_parquet(out / "test_predictions.parquet", index=False)
    h = summary.set_index("model").loc["H_raw_alpha5"]
    gains = []
    for _, row in summary.iterrows():
        if row["model"] == "strict_features":
            continue
        gains.append(
            {
                "model": row["model"],
                "auc_pr_gain_vs_H5": row["auc_pr"] - h["auc_pr"],
                "p_at_0.01_gain_vs_H5": row["p_at_0.01"] - h["p_at_0.01"],
                "p_at_0.05_gain_vs_H5": row["p_at_0.05"] - h["p_at_0.05"],
                "accepted_vs_H5": bool(row["model"] != "H_raw_alpha5" and row["auc_pr"] > h["auc_pr"] and row["p_at_0.01"] >= h["p_at_0.01"]),
            }
        )
    gains_df = pd.DataFrame(gains)
    gains_df.to_csv(out / "gains_vs_H5.csv", index=False)
    audit = {
        "purpose": "combine strongest raw alpha=5 history with exact local Green fields from cached strict block-causal runs",
        "no_wallet_labels": True,
        "unknown_class_3_excluded": True,
        "aggregate_features_excluded": True,
        "runtime_sec": time.time() - started,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Elliptic++ exact Green combo check",
                "",
                "Fair improvement check: keep the strong raw released-history alpha=5 baseline and test whether exact local Green fields add signal.",
                "",
                "## Summary",
                "",
                markdown_table(summary),
                "",
                "## Gains vs H_raw_alpha5",
                "",
                markdown_table(gains_df),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(gains_df.to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
