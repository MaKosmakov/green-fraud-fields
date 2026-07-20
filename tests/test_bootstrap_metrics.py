import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_green_final_bootstrap_ci import (  # noqa: E402
    BUDGETS,
    precision_at,
    prepare_weighted_ranking,
    weighted_rank_metrics,
)


def test_weighted_bootstrap_metrics_match_explicit_chronological_duplication():
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=int)
    scores = np.array([0.9, 0.9, 0.7, 0.7, 0.2, 0.2], dtype=float)
    row_blocks = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    block_counts = np.array([2, 0, 3], dtype=int)

    weighted = weighted_rank_metrics(
        prepare_weighted_ranking(labels, scores, row_blocks),
        block_counts,
    )

    multiplicities = block_counts[row_blocks]
    explicit_labels = np.repeat(labels, multiplicities)
    explicit_scores = np.repeat(scores, multiplicities)
    assert np.isclose(weighted["auc_pr"], average_precision_score(explicit_labels, explicit_scores))
    for budget in BUDGETS:
        assert np.isclose(
            weighted[f"precision_at_{budget:g}"],
            precision_at(explicit_labels, explicit_scores, budget),
        )
