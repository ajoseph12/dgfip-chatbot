"""Per-query retrieval metrics (single relevant fiche per query)."""


def reciprocal_rank(ranked: list[int], target: int) -> float:
    """1 / rank of the target fiche in the ranked list (0 if absent)."""
    for i, fiche_id in enumerate(ranked, 1):
        if fiche_id == target:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked: list[int], target: int, k: int) -> float:
    """1.0 if the target fiche is within the top k, else 0.0."""
    return 1.0 if target in ranked[:k] else 0.0
