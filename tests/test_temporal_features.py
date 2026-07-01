import numpy as np

from green_fraud_fields.ieee_cis import TypedEdge
from green_fraud_fields.temporal_features import CausalFeatureBuilder


def edge(layer, a, b, time):
    return TypedEdge(layer, a, b, 1.0, time, time)


def test_current_row_is_not_seen_before_insertion():
    builder = CausalFeatureBuilder()
    first = builder.process_edges([edge("card1--addr1", "a", "b", 1)], 10.0)
    second = builder.process_edges([edge("card1--addr1", "a", "b", 2)], 10.0)
    assert first["card1__addr1__edge_seen"] == 0
    assert second["card1__addr1__edge_seen"] == 1
    assert second["card1__addr1__count_ab"] == 1
    assert second["card1__addr1__time_since_edge"] == 1


def test_layer_specific_cold_start():
    builder = CausalFeatureBuilder()
    builder.process_edges([edge("card1--addr1", "card", "addr", 1)], 10.0)
    result = builder.process_edges([edge("card1--DeviceInfo", "card", "device", 2)], 10.0)
    assert result["card1__DeviceInfo__cohort"] == "C11"


def test_studentized_feature_uses_only_past_amounts():
    builder = CausalFeatureBuilder(epsilon=1e-6)
    builder.process_edges([edge("x--y", "a", "b", 1)], 9.0)
    result = builder.process_edges([edge("x--y", "a", "b", 2)], 99.0)
    assert np.isclose(result["x__y__delta_log_amt_mean"], 0.0)
    assert np.isclose(result["x__y__t_edge_amount"], 0.0)


def test_recent_burst_increases_velocity():
    builder = CausalFeatureBuilder(short_window=10, long_window=100)
    for t in (1, 2, 3, 4):
        builder.process_edges([edge("x--y", "a", f"b{t}", t)], 1.0)
    result = builder.process_edges([edge("x--y", "a", "z", 5)], 1.0)
    assert result["x__y__count_velocity"] > 0


