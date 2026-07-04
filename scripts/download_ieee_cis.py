from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/ieee_cis")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", "ieee-fraud-detection", "-p", str(out_dir),
    ]
    subprocess.run(command, check=True)
    archives = list(out_dir.glob("*.zip"))
    if not archives:
        raise RuntimeError("Kaggle completed but no archive was found.")
    import zipfile
    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            wanted = {
                "train_transaction.csv",
                "train_identity.csv",
            }
            for member in zipped.namelist():
                if Path(member).name in wanted:
                    zipped.extract(member, out_dir)
                    extracted = out_dir / member
                    target = out_dir / Path(member).name
                    if extracted != target:
                        extracted.replace(target)
    print(f"IEEE-CIS training files are in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
