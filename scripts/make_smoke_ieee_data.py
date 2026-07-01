from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed/ieee_smoke")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.rows
    time = np.cumsum(rng.integers(10, 300, n))
    card = rng.integers(1000, 1080, n)
    addr = rng.integers(100, 135, n)
    device = rng.choice(["iOS", "Android", "Windows", None], n, p=[.3, .35, .25, .1])
    email = rng.choice(["gmail.com", "yahoo.com", "outlook.com", None], n)
    amount = rng.lognormal(3.5, 1.0, n)
    burst = (card % 17 == 0) & (amount > np.quantile(amount, .75))
    new_pair = ((card + addr) % 23 == 0)
    probability = np.clip(.015 + .20 * burst + .10 * new_pair, 0, .8)
    fraud = rng.binomial(1, probability)
    transaction = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "TransactionDT": time,
        "TransactionAmt": amount,
        "isFraud": fraud,
        "card1": card,
        "addr1": addr,
        "P_emaildomain": email,
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        "ProductCD": rng.choice(list("WHCRS"), n),
        "card4": rng.choice(["visa", "mastercard"], n),
        "card6": rng.choice(["credit", "debit"], n),
    })
    identity = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "DeviceInfo": device,
        "DeviceType": np.where(pd.isna(device), None, "desktop"),
        "id_12": rng.choice(["Found", "NotFound", None], n),
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    transaction.to_csv(out / "train_transaction.csv", index=False)
    identity.to_csv(out / "train_identity.csv", index=False)
    print(f"Wrote {n} rows with {int(fraud.sum())} fraud labels to {out.resolve()}")


if __name__ == "__main__":
    main()


