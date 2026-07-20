import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from green_fraud_fields.green_risk_field import (
    GreenRiskFieldBuilder,
    BoundedGreenRiskFieldBuilder,
    corrected_transfer,
    select_by_validation,
)
from green_fraud_fields.laplacian_features import LayerGraph
from green_fraud_fields.modeling import alert_metrics, make_preprocessor, pointwise_tail_score
from green_fraud_fields.temporal_features import CausalFeatureBuilder


def row(transaction_id, time, label, card=1, addr=10):
    return pd.Series({
        "TransactionID": transaction_id,
        "TransactionDT": time,
        "TransactionAmt": 10.0,
        "isFraud": label,
        "card1": card,
        "addr1": addr,
        "P_emaildomain": np.nan,
        "DeviceInfo": np.nan,
    })


def test_pointwise_tail_score_is_prefix_invariant():
    base = np.array([0.1, 0.8, 0.6, 0.9])
    second = np.array([0.2, 0.3, 0.7, 0.4])
    cutoff = 0.7
    full = pointwise_tail_score(base, second, cutoff)
    prefix = pointwise_tail_score(base[:2], second[:2], cutoff)
    np.testing.assert_allclose(full[:2], prefix)
    assert full[1] > full[2]


def test_alert_metrics_break_score_ties_by_arrival_order():
    labels = np.array([1, 0, 0, 0])
    tied_scores = np.ones(4)
    metrics = alert_metrics(labels, tied_scores, 0.25)
    assert metrics["precision_at_0.25"] == 1.0


def test_preprocessor_accepts_pandas_string_columns():
    frame = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "device": ["ios", "web", "ios"]})
    transformed = make_preprocessor(frame).fit_transform(frame)
    assert transformed.shape[0] == len(frame)


def test_delayed_history_never_reads_future_label():
    builder = GreenRiskFieldBuilder(delays_days=(1,), alphas=(5,), ego_radius=1)
    builder.process_row(row(1, 0, 1))
    before = builder.process_row(row(2, 86399, 0))
    assert before["card1__addr1__d1_a5__H_left_exposure_log"] == 0
    at_release_time = builder.process_row(row(3, 86400, 0))
    assert at_release_time["card1__addr1__d1_a5__H_left_exposure_log"] == 0
    after = builder.process_row(row(4, 86401, 0))
    assert after["card1__addr1__d1_a5__H_left_exposure_log"] > 0


def test_new_node_uses_causal_prior_and_missing_flag():
    builder = GreenRiskFieldBuilder(delays_days=(0,), alphas=(5,), ego_radius=1)
    result = builder.process_row(row(1, 0, 1, card=99, addr=77))
    assert result["card1__addr1__d0_a5__H_left"] == 0
    assert result["card1__addr1__d0_a5__H_left_missing"] == 1


def test_candidate_transaction_is_excluded_from_its_own_features():
    green = GreenRiskFieldBuilder(delays_days=(0,), alphas=(5,), ego_radius=1)
    first_green = green.process_row(row(1, 0, 1, card=99, addr=77))
    assert first_green["card1__addr1__d0_a5__H_left_exposure_log"] == 0
    assert first_green["card1__addr1__d0_a5__H_right_exposure_log"] == 0

    second_green = green.process_row(row(2, 1, 0, card=99, addr=77))
    assert second_green["card1__addr1__d0_a5__H_left_exposure_log"] > 0
    assert second_green["card1__addr1__d0_a5__H_right_exposure_log"] > 0

    causal = CausalFeatureBuilder(compute_laplacian=False)
    first_causal = causal.transform(pd.DataFrame([row(1, 0, 1, card=99, addr=77)]))
    assert int(first_causal.loc[0, "card1__addr1__count_a"]) == 0
    assert int(first_causal.loc[0, "card1__addr1__count_b"]) == 0
    assert int(first_causal.loc[0, "card1__addr1__edge_seen"]) == 0


def test_same_timestamp_edges_are_scored_from_same_frozen_state():
    frame = pd.DataFrame([
        row(1, 100, 0, card=1, addr=10),
        row(2, 100, 0, card=1, addr=11),
        row(3, 101, 0, card=1, addr=12),
    ])
    features = CausalFeatureBuilder(compute_laplacian=False).transform(frame)

    assert int(features.loc[0, "card1__addr1__count_a"]) == 0
    assert int(features.loc[1, "card1__addr1__count_a"]) == 0
    assert int(features.loc[1, "card1__addr1__a_seen"]) == 0
    assert int(features.loc[2, "card1__addr1__count_a"]) == 2
    assert int(features.loc[2, "card1__addr1__a_seen"]) == 1


def test_same_timestamp_label_is_excluded_at_delay_zero():
    frame = pd.DataFrame([
        row(1, 100, 1, card=1, addr=10),
        row(2, 100, 0, card=1, addr=10),
        row(3, 101, 0, card=1, addr=10),
    ])
    features = GreenRiskFieldBuilder(delays_days=(0,), alphas=(5,), ego_radius=1).transform(frame)

    assert features.loc[0, "card1__addr1__d0_a5__H_left_exposure_log"] == 0
    assert features.loc[1, "card1__addr1__d0_a5__H_left_exposure_log"] == 0
    assert features.loc[1, "card1__addr1__d0_a5__H_right_exposure_log"] == 0
    assert features.loc[2, "card1__addr1__d0_a5__H_left_exposure_log"] > 0
    assert features.loc[2, "card1__addr1__d0_a5__H_right_exposure_log"] > 0


def test_prior_timestamp_edge_and_label_are_included_when_released():
    frame = pd.DataFrame([
        row(1, 100, 1, card=1, addr=10),
        row(2, 101, 0, card=1, addr=10),
    ])
    causal = CausalFeatureBuilder(compute_laplacian=False).transform(frame)
    green = GreenRiskFieldBuilder(delays_days=(0,), alphas=(5,), ego_radius=1).transform(frame)

    assert int(causal.loc[1, "card1__addr1__edge_seen"]) == 1
    assert int(causal.loc[1, "card1__addr1__count_a"]) == 1
    assert green.loc[1, "card1__addr1__d0_a5__H_left_exposure_log"] > 0
    assert green.loc[1, "card1__addr1__d0_a5__H_right_exposure_log"] > 0


def test_green_feature_parquet_has_block_causal_metadata(tmp_path):
    frame = pd.DataFrame([
        row(1, 100, 1, card=1, addr=10),
        row(2, 101, 0, card=1, addr=10),
    ])
    path = tmp_path / "green_features.parquet"
    GreenRiskFieldBuilder(delays_days=(0,), alphas=(5,), ego_radius=1).write_parquet(frame, str(path), batch_size=1)
    metadata = pq.ParquetFile(path).schema_arrow.metadata
    assert metadata[b"causal_policy"] == b"strict_timestamp_block"
    assert metadata[b"same_timestamp_policy"] == b"block_frozen_t_minus"
    assert metadata[b"label_release_policy"] == b"release_time_strictly_less_than_candidate_timestamp"

    old_path = tmp_path / "old_features.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(old_path, index=False)
    old_metadata = pq.ParquetFile(old_path).schema_arrow.metadata or {}
    assert old_metadata.get(b"causal_policy") != b"strict_timestamp_block"


def test_green_smoothed_field_matches_two_node_direct_solve():
    graph = LayerGraph()
    graph.add_edge("a", "b", 1.0)
    nodes, matrix, index = graph.ego_system("a", "b", 0.5, 2, 10)
    signal = np.array([1.0 if node == "a" else 0.0 for node in nodes])
    actual = 0.5 * np.linalg.solve(matrix.toarray(), signal)
    expected = 0.5 * np.linalg.solve(
        np.array([[1.5, -1.0], [-1.0, 1.5]]), np.array([1.0, 0.0])
    )
    assert np.allclose(
        [actual[index["a"]], actual[index["b"]]], expected
    )


def test_corrected_transfer_matches_explicit_inverse_update():
    lambda_ = 0.4
    old_a = np.array([[1.4, -1.0], [-1.0, 1.4]])
    green = np.linalg.inv(old_a)
    signal = np.array([0.8, 0.2])
    shat = lambda_ * green @ signal
    b = np.array([1.0, -1.0])
    resistance = b @ green @ b
    left, right = corrected_transfer(
        shat[0], shat[1], green[0, 0], green[1, 1], green[0, 1], resistance
    )
    updated = lambda_ * np.linalg.inv(old_a + np.outer(b, b)) @ signal
    assert np.allclose(updated - shat, [left, right])


def test_builder_is_deterministic():
    frame = pd.DataFrame([
        row(1, 0, 1),
        row(2, 10, 0, card=1, addr=11),
        row(3, 20, 0, card=2, addr=10),
    ])
    kwargs = dict(delays_days=(0, 1), alphas=(5,), ego_radius=1, max_ego_nodes=20)
    first = GreenRiskFieldBuilder(**kwargs).transform(frame)
    second = GreenRiskFieldBuilder(**kwargs).transform(frame)
    pd.testing.assert_frame_equal(first, second)


def test_green_builder_uses_dense_path_for_small_egos():
    frame = pd.DataFrame([
        row(1, 0, 1),
        row(2, 10, 0, card=1, addr=11),
        row(3, 20, 0, card=2, addr=10),
    ])
    builder = GreenRiskFieldBuilder(
        delays_days=(0,), alphas=(5,), ego_radius=1, max_ego_nodes=20,
        dense_threshold=150,
    )
    builder.transform(frame)
    summary = builder.runtime_summary()
    assert summary["dense_path_count"] > 0
    assert summary["sparse_path_count"] == 0
    assert summary["dense_assembly_seconds"] >= 0
    assert summary["dense_solve_seconds"] >= 0


def test_validation_selection_ignores_test_outcome():
    candidates = [
        (0.7, "alpha5", 0.1),
        (0.6, "alpha20", 0.99),
    ]
    assert select_by_validation(candidates)[1] == "alpha5"


def test_bounded_builder_sets_fallback_flags():
    frame = pd.DataFrame([
        row(1, 0, 1, card=1, addr=10),
        row(2, 86400, 0, card=1, addr=11),
        row(3, 86401, 0, card=1, addr=12),
    ])
    builder = BoundedGreenRiskFieldBuilder(exact_cap=2, fallback_cap=1)
    result = builder.transform(frame)
    assert "agg__d1_a5__S_ego_over_cap" in result.columns
    assert result["agg__d1_a5__S_ego_over_cap"].fillna(0).max() >= 1

