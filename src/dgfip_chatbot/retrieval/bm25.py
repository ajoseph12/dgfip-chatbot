"""BM25 (lexical) retriever — the keyword baseline. Scores chunks by term overlap over the
pre-normalized `text_lexical` column, then max-pools to fiches like the dense retriever."""

import pandas as pd
from rank_bm25 import BM25Okapi

from dgfip_chatbot.config import settings
from dgfip_chatbot.data.clean import lexical_normalize
from dgfip_chatbot.retrieval.base import RetrievalResult, Retriever


class BM25Retriever(Retriever):
    def __init__(self, meta: pd.DataFrame | None = None):
        # Use an in-memory chunk table if given (experiments); else load from disk.
        if meta is None:
            if not settings.chunks_path.exists():
                raise FileNotFoundError("chunks not found — run `make data` first")
            meta = pd.read_parquet(settings.chunks_path)
        self.meta = meta.reset_index(drop=True)
        # `text_lexical` is already lowercase/accent-folded/stopword-stripped → split on space.
        corpus = [t.split() for t in self.meta["text_lexical"]]
        self.bm25 = BM25Okapi(corpus, k1=settings.bm25_k1, b=settings.bm25_b)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k or settings.top_k
        # Normalize the query the same way the corpus was, then score every chunk.
        tokens = lexical_normalize(query).split()
        scores = self.bm25.get_scores(tokens)  # one score per chunk

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
