"""Phase 3 integration tests — BM25, hybrid, and the harness. Skipped unless the ml group
is installed AND the index is built (`uv sync --group ml && make index`)."""

import pandas as pd
import pytest

from dgfip_chatbot.config import settings

pytest.importorskip("rank_bm25")
pytest.importorskip("sentence_transformers")
if not settings.embeddings_path.exists():
    pytest.skip("index not built — run `make index`", allow_module_level=True)

from dgfip_chatbot.eval.harness import evaluate  # noqa: E402
from dgfip_chatbot.retrieval.bm25 import BM25Retriever  # noqa: E402
from dgfip_chatbot.retrieval.dense import DenseRetriever  # noqa: E402
from dgfip_chatbot.retrieval.hybrid import HybridRetriever  # noqa: E402


@pytest.fixture(scope="module")
def retrievers():
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    return dense, bm25, HybridRetriever(dense=dense, bm25=bm25)


def test_bm25_and_hybrid_return_ranked_unique_fiches(retrievers):
    _, bm25, hybrid = retrievers
    for r in (bm25, hybrid):
        res = r.retrieve("Comment déclarer mes revenus ?", k=5)
        assert 1 <= len(res) <= 5
        ids = [x.fiche_id for x in res]
        assert len(ids) == len(set(ids))  # unique fiches
        scores = [x.score for x in res]
        assert scores == sorted(scores, reverse=True)  # descending


def test_evaluate_returns_expected_metrics(retrievers):
    dense, _, _ = retrievers
    q = pd.DataFrame(
        {
            "question": ["Comment déclarer mes revenus en ligne ?"],
            "fiche_id": [2],
            "cat": ["declarer-mes-revenus"],
        }
    )
    out = evaluate(dense, q, ks=[1, 5])
    assert set(out) == {"overall", "per_theme", "macro"}
    assert {"MRR", "hit@1", "hit@5"} <= set(out["overall"].index)
    assert 0.0 <= out["overall"]["hit@5"] <= 1.0
