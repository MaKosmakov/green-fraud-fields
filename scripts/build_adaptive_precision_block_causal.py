from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from green_fraud_fields.ieee_cis import load_ieee_cis_cached
from green_fraud_fields.modeling import save_json
from run_green_adaptive_theory import PrecisionWeightedGreenBuilder, parse_ints


def has_block_causal_metadata(path: Path, delay: int) -> bool:
    if not path.exists():
        return False
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return (
        metadata.get(b"causal_policy") == b"strict_timestamp_block"
        and metadata.get(b"same_timestamp_policy") == b"block_frozen_t_minus"
        and metadata.get(b"label_release_policy") == b"release_time_strictly_less_than_candidate_timestamp"
        and metadata.get(b"delay_days") == str(delay).encode()
    )


def run_window(args: argparse.Namespace, window: int) -> dict:
    started = time.time()
    out = Path(args.out_dir) / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "precision_weighted_green.parquet"
    if path.exists() and not has_block_causal_metadata(path, args.delay):
        path.unlink()

    data = load_ieee_cis_cached(
        args.data_dir,
        (window + 1) * args.window_size,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    )
    data = data.iloc[window * args.window_size : (window + 1) * args.window_size].reset_index(drop=True)
    if data.empty:
        payload = {
            "window": window,
            "seconds": time.time() - started,
            "rows": 0,
            "skipped": True,
            "reason": "window outside available data",
            "delay": args.delay,
            "alpha": args.alpha,
            "precision_alpha": args.precision_alpha,
            "causal_policy": "strict_timestamp_block",
            "same_timestamp_policy": "block_frozen_t_minus",
            "label_release_policy": "release_time_strictly_less_than_candidate_timestamp",
            "runtime": {"skipped": True},
        }
        save_json(out / "precision_feature_runtime.json", payload)
        print(json.dumps(payload, indent=2), flush=True)
        return payload

    if path.exists() and has_block_causal_metadata(path, args.delay) and not args.force:
        runtime = {"cached": True, "path": str(path)}
    else:
        builder = PrecisionWeightedGreenBuilder(
            delay_days=args.delay,
            history_alpha=args.alpha,
            precision_alpha=args.precision_alpha,
            radius=args.ego_radius,
            cap=args.max_ego_nodes,
        )
        runtime = builder.write_parquet(data, path, batch_size=args.batch_size)
        runtime["cached"] = False
        runtime["path"] = str(path)

    payload = {
        "window": window,
        "seconds": time.time() - started,
        "delay": args.delay,
        "alpha": args.alpha,
        "precision_alpha": args.precision_alpha,
        "causal_policy": "strict_timestamp_block",
        "same_timestamp_policy": "block_frozen_t_minus",
        "label_release_policy": "release_time_strictly_less_than_candidate_timestamp",
        "runtime": runtime,
    }
    save_json(out / "precision_feature_runtime.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/00_adaptive_precision_block_causal")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--precision-alpha", type=float, default=5.0)
    parser.add_argument("--ego-radius", type=int, default=2)
    parser.add_argument("--max-ego-nodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-data-cache", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    runtimes = {}
    for window in parse_ints(args.windows):
        payload = run_window(args, window)
        runtimes[f"window_{window}"] = payload
        rows.append(
            {
                "window": window,
                "runtime_seconds": payload["seconds"],
                "delay": payload["delay"],
                "cached": payload["runtime"].get("cached"),
                "edge_solves": payload["runtime"].get("edge_solves"),
                "mean_ego_size": payload["runtime"].get("mean_ego_size"),
                "max_ego_size": payload["runtime"].get("max_ego_size"),
            }
        )
    pd.DataFrame(rows).to_csv(root / "precision_feature_window_summary.csv", index=False)
    save_json(root / "precision_feature_runtime.json", runtimes)


if __name__ == "__main__":
    main()
