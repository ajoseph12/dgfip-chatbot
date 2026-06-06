"""Pure unit tests for the retrieval metrics (no model / index needed)."""

from dgfip_chatbot.eval.metrics import hit_at_k, reciprocal_rank


def test_reciprocal_rank():
    assert reciprocal_rank([5, 9, 2], 5) == 1.0  # rank 1
    assert reciprocal_rank([9, 5, 2], 5) == 0.5  # rank 2
    assert reciprocal_rank([9, 2], 5) == 0.0  # absent


def test_hit_at_k():
    ranked = [9, 5, 2, 7]
    assert hit_at_k(ranked, 5, 1) == 0.0  # 5 is at rank 2, not in top 1
    assert hit_at_k(ranked, 5, 2) == 1.0
    assert hit_at_k(ranked, 7, 3) == 0.0
    assert hit_at_k(ranked, 7, 4) == 1.0
