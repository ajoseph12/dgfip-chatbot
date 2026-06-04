"""Phase 2 tests — dense retrieval. Skipped unless the ml group is installed AND the
index has been built (`uv sync --group ml && make index`)."""

import numpy as np
import pytest

from dgfip_chatbot.config import settings

pytest.importorskip("sentence_transformers")  # needs `uv sync --group ml`
if not settings.embeddings_path.exists():
    pytest.skip("index not built — run `make index`", allow_module_level=True)

from dgfip_chatbot.retrieval.dense import DenseRetriever  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    return DenseRetriever()


def test_embeddings_are_unit_norm_and_right_shape():
    emb = np.load(settings.embeddings_path)
    assert emb.shape[0] > 0 and emb.shape[1] == 768  # e5-base dimension
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)  # L2-normalized rows


def test_retrieve_returns_ranked_unique_fiches(retriever):
    res = retriever.retrieve("Comment déclarer mes revenus ?", k=5)
    assert 1 <= len(res) <= 5
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)  # descending
    ids = [r.fiche_id for r in res]
    assert len(ids) == len(set(ids))  # one entry per fiche (max-pool)
    assert all(0 <= r.fiche_id <= 112 for r in res)
    assert all(-1.001 <= r.score <= 1.001 for r in res)  # cosine range


def test_self_retrieval_surfaces_own_fiche(retriever):
    # Querying with a chunk's own text should surface its fiche near the top.
    sample = retriever.meta.sample(5, random_state=0)
    for row in sample.itertuples(index=False):
        hits = {r.fiche_id for r in retriever.retrieve(row.text, k=3)}
        assert row.fiche_id in hits
