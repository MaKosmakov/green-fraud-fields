from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the frozen result files and source paths recorded in the experiment manifest."
    )
    parser.add_argument(
        "--manifest",
        default="manifests/ieee_green_block_causal_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/ieee_green_final_review_gates_v2_block_causal",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    verified = 0
    for record in manifest.get("result_files", []):
        relative = record["relative_to_output_root"]
        path = output_root / relative
        if not path.is_file():
            failures.append(f"missing result: {path}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            failures.append(f"size mismatch: {path}")
            continue
        if sha256(path) != record["sha256"]:
            failures.append(f"SHA-256 mismatch: {path}")
            continue
        verified += 1

    for relative in manifest.get("source_scripts", []):
        if not Path(relative).is_file():
            failures.append(f"missing source script: {relative}")

    if failures:
        print("Manifest verification FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Manifest verification PASS: {verified} result files and all source scripts match.")


if __name__ == "__main__":
    main()
