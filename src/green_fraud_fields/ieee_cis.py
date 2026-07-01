from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ENTITY_COLUMNS = [
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "ProductCD",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
    "id_19", "id_20", "id_28", "id_29", "id_30", "id_31", "id_32",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

RELATIONS = [
    ("card1", "addr1"),
    ("card1", "DeviceInfo"),
    ("card1", "P_emaildomain"),
    ("card1", "R_emaildomain"),
    ("card1", "ProductCD"),
    ("DeviceInfo", "P_emaildomain"),
    ("DeviceInfo", "addr1"),
    ("addr1", "P_emaildomain"),
]


@dataclass(frozen=True)
class TypedEdge:
    layer: str
    a: str
    b: str
    w: float
    transaction_id: Any
    time: float


def _entity(column: str, value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return f"{column}={value}"


def build_transaction_edges(row: pd.Series | dict[str, Any]) -> list[TypedEdge]:
    edges: list[TypedEdge] = []
    for left, right in RELATIONS:
        if left not in row or right not in row:
            continue
        a, b = _entity(left, row[left]), _entity(right, row[right])
        if a is None or b is None:
            continue
        edges.append(
            TypedEdge(
                layer=f"{left}--{right}",
                a=a,
                b=b,
                w=1.0,
                transaction_id=row["TransactionID"],
                time=float(row["TransactionDT"]),
            )
        )
    return edges


def transaction_edge_records(
    df: pd.DataFrame,
    *,
    include_amount: bool = False,
    include_label: bool = False,
) -> list[dict[str, Any]]:
    """Return only the columns needed for causal edge/history feature builders.

    Iterating over full IEEE-CIS rows is expensive because the frame contains many
    Arrow-backed string columns. The graph/history builders only need relation
    endpoints plus transaction id/time and, depending on the caller, amount/label.
    """
    required = {"TransactionID", "TransactionDT"}
    for left, right in RELATIONS:
        required.add(left)
        required.add(right)
    if include_amount:
        required.add("TransactionAmt")
    if include_label:
        required.add("isFraud")
    columns = [column for column in df.columns if column in required]
    return df[columns].to_dict("records")


def load_ieee_cis(data_dir: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    tx_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing {tx_path}")
    tx = pd.read_csv(tx_path, nrows=max_rows, low_memory=False)
    if identity_path.exists():
        identity = pd.read_csv(identity_path, low_memory=False)
        tx = tx.merge(identity, how="left", on="TransactionID", validate="one_to_one")
    required = {"TransactionID", "TransactionDT", "isFraud"}
    missing = required - set(tx.columns)
    if missing:
        raise ValueError(f"IEEE-CIS data is missing columns: {sorted(missing)}")
    tx["_source_order"] = np.arange(len(tx), dtype=np.int64)
    return tx.sort_values(["TransactionDT", "_source_order"], kind="stable").reset_index(drop=True)


def load_ieee_cis_cached(
    data_dir: str | Path,
    max_rows: int | None = None,
    cache_dir: str | Path = "data/processed",
    force_cache: bool = False,
) -> pd.DataFrame:
    """Load IEEE-CIS from a processed parquet cache.

    The raw Kaggle files are relatively expensive to read and merge repeatedly,
    especially for rolling-window experiments. This helper materializes the
    chronological, identity-merged training table once, then serves future
    window slices from parquet.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ieee_cis_train_chronological.parquet"
    meta_path = cache_dir / "ieee_cis_train_chronological.meta.json"
    tx_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"

    def raw_mtime(path: Path) -> float | None:
        return path.stat().st_mtime if path.exists() else None

    expected_meta = {
        "train_transaction_mtime": raw_mtime(tx_path),
        "train_identity_mtime": raw_mtime(identity_path),
        "cache_version": 1,
    }
    cache_ok = cache_path.exists() and not force_cache
    if cache_ok and meta_path.exists():
        import json

        try:
            cache_ok = json.loads(meta_path.read_text(encoding="utf-8")) == expected_meta
        except json.JSONDecodeError:
            cache_ok = False
    elif cache_ok:
        cache_ok = False

    if not cache_ok:
        full = load_ieee_cis(data_dir, max_rows=None)
        full.to_parquet(cache_path, index=False, compression="zstd")
        import json

        meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")

    data = pd.read_parquet(cache_path)
    if max_rows is not None:
        data = data.iloc[:max_rows]
    return data.reset_index(drop=True)


def tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    amount = pd.to_numeric(df.get("TransactionAmt"), errors="coerce")
    out["TransactionAmt"] = amount
    out["log_TransactionAmt"] = np.log1p(amount.clip(lower=0))
    seconds = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0)
    hour = (seconds / 3600.0) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["relative_day"] = np.floor(seconds / 86400)
    for col in ["ProductCD", "card4", "card6"]:
        if col in df:
            out[col] = df[col].astype("string").fillna("__MISSING__")
    identity_present = pd.Series(False, index=df.index)
    for col in [c for c in ENTITY_COLUMNS if c in df]:
        if col.startswith("id_") or col in {"DeviceType", "DeviceInfo"}:
            out[f"{col}_missing"] = df[col].isna().astype("int8")
            identity_present |= df[col].notna()
    out["identity_present"] = identity_present.astype("int8")
    return out


def chronological_split(n: int, train_fraction: float = 0.6, valid_fraction: float = 0.2):
    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + valid_fraction))
    return slice(0, train_end), slice(train_end, valid_end), slice(valid_end, n)

