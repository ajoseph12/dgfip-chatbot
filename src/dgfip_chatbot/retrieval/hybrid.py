"""Hybrid retriever — fuses dense + BM25 with confidence-aware **score fusion**.

Each retriever's per-fiche scores are min-max normalized per query, then combined as a
weighted sum: ``w · dense_norm + (1-w) · bm25_norm`` (balanced 0.5/0.5 by default). Score-based
fusion beat rank-based RRF on dev because it preserves each retriever's *confidence* — a
fiche one retriever is very sure about wins even if the other disagrees (see
reports/fusion_stemming.md).
"""

from dgfip_chatbot.config import settings
from dgfip_chatbot.retrieval.base import RetrievalResult, Retriever

_FULL = 1_000_000  # ask each sub-retriever for its full fiche ranking


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Scale a query's fiche scores to [0, 1] so dense (cosine) and BM25 are comparable."""
    vals = scores.values()
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1e-9
    return {f: (s - lo) / rng for f, s in scores.items()}


class HybridRetriever(Retriever):
    def __init__(self, dense: Retriever, bm25: Retriever, dense_weight: float | None = None):
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = (
            dense_weight if dense_weight is not None else settings.fusion_dense_weight
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k or settings.top_k
        # retrieve top k fiches using each method
        dense_hits = self.dense.retrieve(query, k=_FULL)
        bm25_hits = self.bm25.retrieve(query, k=_FULL)
        # min-max normalise
        dn = _minmax({h.fiche_id: h.score for h in dense_hits})
        bn = _minmax({h.fiche_id: h.score for h in bm25_hits})
        # metadata/snippet per fiche (prefer the dense best-chunk snippet)
        info = {h.fiche_id: h for h in bm25_hits}
        info.update({h.fiche_id: h for h in dense_hits})

        w = self.dense_weight

        def fused(fiche_id: int) -> float:
            return w * dn.get(fiche_id, 0.0) + (1.0 - w) * bn.get(fiche_id, 0.0)

        ranked = sorted(info, key=fused, reverse=True)[:k]
        return [
            RetrievalResult(
                fiche_id=fid,
                score=fused(fid),
                titre=info[fid].titre,
                url=info[fid].url,
                snippet=info[fid].snippet,
            )
            for fid in ranked
        ]
