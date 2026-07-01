import numpy as np

from green_fraud_fields.laplacian_features import LayerGraph


def test_two_isolated_nodes_have_expected_resistance():
    feature = LayerGraph().features("a", "b", lambda_=2.0)
    assert np.isclose(feature["R_ab"], 1.0)
    assert np.isclose(feature["G_ab"], 0.0)


def test_connected_edge_reduces_resistance_and_identity_holds():
    graph = LayerGraph()
    before = graph.features("a", "b", lambda_=1.0)
    graph.add_edge("a", "b")
    after = graph.features("a", "b", lambda_=1.0)
    assert after["R_ab"] < before["R_ab"]
    assert after["G_ab"] > 0
    assert np.isclose(
        after["R_ab"], after["G_aa"] + after["G_bb"] - 2 * after["G_ab"]
    )
    assert np.isclose(after["logdet_edge"], np.log1p(after["R_ab"]))


def test_one_isolated_endpoint_has_zero_cross_term():
    graph = LayerGraph()
    graph.add_edge("b", "c")
    feature = graph.features("a", "b", lambda_=1.0)
    assert np.isclose(feature["G_ab"], 0.0)


def test_dense_and_sparse_matrices_match_on_small_graph():
    graph = LayerGraph()
    for u, v, w in [
        ("a", "b", 1.0),
        ("a", "c", 2.0),
        ("b", "c", 0.5),
        ("c", "d", 3.0),
        ("b", "d", 1.5),
    ]:
        graph.add_edge(u, v, w)
    nodes = graph.ego_nodes(("a", "d"), radius=2, max_nodes=10)
    dense, dense_index, dense_timing = graph.dense_matrix_for_nodes_timed(nodes, 0.7)
    sparse, sparse_index, sparse_timing = graph.matrix_for_nodes_timed(nodes, 0.7)
    assert dense_timing["path"] == "dense"
    assert sparse_timing["path"] == "sparse"
    assert dense_index == sparse_index
    assert np.allclose(dense, sparse.toarray(), atol=1e-10)


def test_dense_and_sparse_endpoint_green_values_match():
    graph = LayerGraph()
    for u, v, w in [
        ("a", "b", 1.0),
        ("a", "c", 2.0),
        ("b", "d", 1.25),
        ("c", "d", 0.75),
        ("d", "e", 1.5),
    ]:
        graph.add_edge(u, v, w)
    nodes = graph.ego_nodes(("a", "e"), radius=3, max_nodes=20)
    dense, dense_index, _ = graph.dense_matrix_for_nodes_timed(nodes, 0.3)
    sparse, sparse_index, _ = graph.matrix_for_nodes_timed(nodes, 0.3)
    rhs = np.zeros((len(nodes), 2))
    rhs[dense_index["a"], 0] = 1.0
    rhs[dense_index["e"], 1] = 1.0
    dense_solution = np.linalg.solve(dense, rhs)
    sparse_solution = np.linalg.solve(sparse.toarray(), rhs)
    assert dense_index == sparse_index
    assert np.allclose(dense_solution, sparse_solution, atol=1e-8)

