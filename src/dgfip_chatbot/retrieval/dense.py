"""Dense retriever: cosine search over cached chunk embeddings, max-pooled to fiches."""

import numpy as np
import pandas as pd

from dgfip_chatbot.config import settings
from dgfip_chatbot.data.clean import canonicalize
from dgfip_chatbot.retrieval.base import RetrievalResult, Retriever
from dgfip_chatbot.retrieval.embedder import Embedder


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder | None = None):
        if not settings.embeddings_path.exists():
            raise FileNotFoundError("embeddings not found — run `make index` first")
        # Cached chunk vectors (already L2-normalized) and their aligned metadata.
        self.embeddings = np.load(settings.embeddings_path)
        self.meta = pd.read_parquet(settings.embeddings_meta_path)
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k or settings.top_k
        # Encode the (cleaned) query; both sides are normalized, so dot product = cosine.
        qv = self.embedder.encode([canonicalize(query)], kind="query")[0]
        scores = self.embeddings @ qv  # one score per chunk

        # Max-pool to fiches: keep each fiche's best-scoring chunk, then rank fiches.
        df = self.meta.assign(score=scores)
        best_per_fiche = df.loc[df.groupby("fiche_id")["score"].idxmax()]
        top = best_per_fiche.sort_values("score", ascending=False).head(k)
        return [
            RetrievalResult(
                fiche_id=int(r.fiche_id),
                score=float(r.score),
                titre=str(r.titre),
                url=str(r.url),
                snippet=str(r.text),
            )
            for r in top.itertuples(index=False)
        ]
