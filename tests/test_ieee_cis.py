import numpy as np
import pandas as pd

from green_fraud_fields.ieee_cis import build_transaction_edges, chronological_split


def test_missing_values_do_not_create_graph_nodes():
    row = pd.Series({
        "TransactionID": 1, "TransactionDT": 0, "card1": 12,
        "addr1": np.nan, "DeviceInfo": np.nan, "P_emaildomain": "gmail.com",
        "R_emaildomain": np.nan, "ProductCD": "W",
    })
    edges = build_transaction_edges(row)
    assert edges
    assert all("nan" not in edge.a.lower() and "nan" not in edge.b.lower() for edge in edges)


def test_chronological_split_is_ordered():
    train, valid, test = chronological_split(100)
    assert train.stop <= valid.start < valid.stop <= test.start


