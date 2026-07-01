from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from green_fraud_fields.ieee_cis import load_ieee_cis_cached
from green_fraud_fields.modeling import save_json
from run_green_moderate_scale import load_or_make_base, load_or_make_green, parse_ints


def run_window(args: argparse.Namespace, window: int) -> dict:
    started = time.time()
    out = Path(args.out_dir) / f"window_{window}"
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"

    def status(phase: str, **extra) -> None:
        save_json(
            status_path,
            {
                "window": window,
                "phase": phase,
                "elapsed_seconds": time.time() - started,
                **extra,
            },
        )

    status("load_data")
    full = load_ieee_cis_cached(
        args.data_dir,
        (window + 1) * args.window_size,
        cache_dir=args.data_cache_dir,
        force_cache=args.force_data_cache,
    )
    data = full.iloc[window * args.window_size : (window + 1) * args.window_size].reset_index(drop=True)

    status("base_features", rows=len(data))
    base_started = time.time()
    base = load_or_make_base(data, out / "base_features.parquet")
    base_seconds = time.time() - base_started

    delays = parse_ints(args.delays)
    status("green_features", rows=len(data), delays=delays, alpha=args.alpha)
    green_started = time.time()
    green_runtime = load_or_make_green(
        data,
        out / "green_features.parquet",
        delays,
        args.alpha,
        args.force_features,
    )
    green_seconds = time.time() - green_started

    runtime = {
        "window": window,
        "seconds": time.time() - started,
        "base_seconds": base_seconds,
        "green_seconds": green_seconds,
        "rows": len(data),
        "base_columns": int(base.shape[1]),
        "delays": delays,
        "alpha": args.alpha,
        "dense_exact_radius": 2,
        "dense_exact_cap": 100,
        "causal_policy": "strict_timestamp_block",
        "same_timestamp_policy": "block_frozen_t_minus",
        "label_release_policy": "release_time_strictly_less_than_candidate_timestamp",
        "random_forest": "not used",
        "green_runtime": green_runtime,
    }
    save_json(out / "feature_runtime.json", runtime)
    status("complete", runtime_seconds=runtime["seconds"])
    print(json.dumps(runtime, indent=2), flush=True)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ieee_cis")
    parser.add_argument("--data-cache-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs/ieee_green_final_review_gates_v2_block_causal/00_moderate_100k_block_causal")
    parser.add_argument("--window-size", type=int, default=100000)
    parser.add_argument("--windows", default="0,1,2,3,4")
    parser.add_argument("--delays", default="0,1,3,7,14")
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--force-data-cache", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    runtimes = {}
    rows = []
    for window in parse_ints(args.windows):
        runtime = run_window(args, window)
        runtimes[f"window_{window}"] = runtime
        rows.append(
            {
                "window": window,
                "runtime_seconds": runtime["seconds"],
                "base_seconds": runtime["base_seconds"],
                "green_seconds": runtime["green_seconds"],
                "rows": runtime["rows"],
                "base_columns": runtime["base_columns"],
                "green_edge_solves": runtime["green_runtime"].get("edge_solves"),
                "green_mean_ego_size": runtime["green_runtime"].get("mean_ego_size"),
                "green_max_ego_size": runtime["green_runtime"].get("max_ego_size"),
            }
        )
    pd.DataFrame(rows).to_csv(root / "feature_window_summary.csv", index=False)
    save_json(root / "feature_runtime.json", runtimes)


if __name__ == "__main__":
    main()

