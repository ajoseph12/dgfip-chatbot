"""`make experiments`: dev-set ablations on the hybrid retriever.

Compares, all on the **dev** split (the test set stays frozen for the final run):
  0. baseline (chunked 256, no title)
  1. + title prepended to every chunk
  2. + semantic chunking (cap 256)
  3. + both

A second entry point, ``fusion_stemming_sweep`` (``make fusion-stemming``), reproduces
``reports/fusion_stemming.md``: RRF vs confidence-aware score-fusion (dense 0.3/0.5/0.7), each
with and without BM25 **stemming**. Stemming is the *rejected* lever (no dev hit@1 gain) — it
is kept here as an isolated, reproducible experiment (mirroring the RRF baseline) and never
touches the production BM25 path (`clean.lexical_normalize` does not stem).

Each variant is built **in memory** (no clobbering the canonical `data/processed` index).
Evaluation is fully vectorized — the dev queries are encoded once, then everything is matrix
math (no per-query model calls or pandas loops), so a variant scores in seconds.
"""

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from dgfip_chatbot.config import settings
from dgfip_chatbot.data.chunk import chunk_text, count_words
from dgfip_chatbot.data.clean import canonicalize, lexical_normalize
from dgfip_chatbot.data.loaders import load_fiches
from dgfip_chatbot.eval.run import _add_cat, _overall_table, _theme_label
from dgfip_chatbot.retrieval.embedder import Embedder

_N_FICHES = 113
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


# --- chunking variants ---------------------------------------------------------------


def _sentence_units(text: str) -> list[str]:
    """Split text into sentence-ish units (on . ! ? or newlines), dropping blanks — these
    are the candidate boundaries semantic chunking decides between."""
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def semantic_chunk(text: str, embedder: Embedder, cap: int, percentile: int = 90) -> list[str]:
    """Cut where adjacent sentences are *least* similar (top `percentile` of embedding
    distance), keeping `cap` (words) as a hard maximum."""
    units = _sentence_units(text)
    if len(units) <= 1:
        return units or [text.strip()]
    embs = embedder.encode(units, kind="passage")  # L2-normalized
    # Cosine distance between adjacent sentence pairs (1 - cos sim); large = topic shift.
    distances = [1.0 - float(embs[i] @ embs[i + 1]) for i in range(len(embs) - 1)]
    # Only the largest gaps (top `percentile`) qualify as cut points.
    thr = float(np.percentile(distances, percentile))

    # Greedily pack sentences into a chunk until the word `cap` or a topic-shift boundary.
    chunks: list[str] = []
    cur: list[str] = [units[0]]
    cur_len = count_words(units[0])
    for i in range(1, len(units)):
        u = units[i]
        ulen = count_words(u)
        # Flush if this sentence would overflow the cap, or the gap before it is a boundary.
        if cur and (cur_len + ulen > cap or distances[i - 1] >= thr):
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(u)
        cur_len += ulen
    if cur:  # flush the trailing chunk
        chunks.append("\n".join(cur))
    return chunks


def build_variant(embedder: Embedder, *, prepend_title: bool, semantic: bool) -> pd.DataFrame:
    """Build a chunk table (text + text_lexical + fiche_id) for one variant, in memory."""
    cap = int(settings.chunk_cap)
    rows: list[dict] = []
    for fiche in load_fiches():
        # If we prepend the title to every chunk, strip the duplicate first-line title.
        clean = canonicalize(fiche.texte, strip_title=prepend_title, title=fiche.titre)
        pieces = (
            semantic_chunk(clean, embedder, cap=cap)
            if semantic
            else chunk_text(clean, cap=cap, overlap=settings.chunk_overlap)
        )
        for piece in pieces:
            body = f"{fiche.titre}\n{piece}" if prepend_title else piece
            rows.append(
                {"fiche_id": fiche.fiche_id, "text": body, "text_lexical": lexical_normalize(body)}
            )
    return pd.DataFrame(rows)


# --- vectorized hybrid evaluation ----------------------------------------------------


def _max_pool(chunk_scores: np.ndarray, chunk_fiche: np.ndarray) -> np.ndarray:
    """[Nq x n_chunks] chunk scores -> [Nq x 113] fiche scores (best chunk per fiche).

    Each fiche is scored by its single strongest chunk — the same max-pool the production
    retrievers use, here vectorized over all queries at once.
    """
    # -inf init so a fiche with no chunks (shouldn't happen; all 113 covered) sorts last.
    out = np.full((chunk_scores.shape[0], _N_FICHES), -np.inf)
    for f in range(_N_FICHES):
        cols = np.where(chunk_fiche == f)[0]  # column indices of this fiche's chunks
        if len(cols):
            out[:, f] = chunk_scores[:, cols].max(axis=1)  # best chunk per query
    return out


def _ranks(scores: np.ndarray) -> np.ndarray:
    """[Nq x F] scores -> [Nq x F] 1-based rank of each fiche (higher score = rank 1)."""
    # `order[q]` lists fiche ids best-first; we want the inverse — each fiche's rank.
    order = np.argsort(-scores, axis=1)  # descending: order[:, 0] is the top fiche per query
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    # Scatter: the fiche sitting at position p gets rank p+1 (inverse permutation of `order`).
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return ranks


def eval_hybrid_fast(
    chunks: pd.DataFrame,
    query_vecs: np.ndarray,
    queries_lex: list[str],
    targets: np.ndarray,
    cats: np.ndarray,
    embedder: Embedder,
    ks: list[int],
    c: int,
) -> dict:
    """Hybrid (RRF of dense + BM25) metrics on a question set, fully vectorized."""
    chunk_emb = embedder.encode(chunks["text"].tolist(), kind="passage")  # [n x d]
    chunk_fiche = chunks["fiche_id"].to_numpy()

    dense_rank = _ranks(_max_pool(query_vecs @ chunk_emb.T, chunk_fiche))

    bm25 = BM25Okapi(
        [t.split() for t in chunks["text_lexical"]], k1=settings.bm25_k1, b=settings.bm25_b
    )
    bm25_chunk = np.vstack([bm25.get_scores(q.split()) for q in queries_lex])  # [Nq x n]
    bm25_rank = _ranks(_max_pool(bm25_chunk, chunk_fiche))

    # RRF: fuse by summed reciprocal rank 1/(c+rank), then re-rank the fused scores.
    fused_rank = _ranks(1.0 / (c + dense_rank) + 1.0 / (c + bm25_rank))
    target_rank = fused_rank[np.arange(len(targets)), targets]  # rank of the true fiche per query

    df = pd.DataFrame({"cat": cats, "MRR": 1.0 / target_rank})
    for k in ks:
        df[f"hit@{k}"] = (target_rank <= k).astype(float)
    cols = ["MRR"] + [f"hit@{k}" for k in ks]
    return {
        "overall": df[cols].mean(),  # micro: every question weighted equally
        "per_theme": df.groupby("cat")[cols].mean(),  # one row per theme
        "macro": df.groupby("cat")[cols].mean().mean(),  # themes equal (unused by callers)
    }


# --- fusion + stemming sweep (reproduces reports/fusion_stemming.md) ------------------

_FR_STEMMER = None  # cached Snowball French stemmer (experiment-only)


def _stem_lexical(text_lexical: str) -> str:
    """Snowball-French-stem an already lexically-normalized token string.

    Isolated to this harness: the production BM25 path (`clean.lexical_normalize`) does **not**
    stem. Stemming was tried and dropped (no dev hit@1 gain); this keeps that rejected
    experiment reproducible, just as the RRF baseline is kept. See reports/fusion_stemming.md.
    """
    global _FR_STEMMER
    if _FR_STEMMER is None:
        from snowballstemmer import stemmer  # experiment-only dep (ml group)

        _FR_STEMMER = stemmer("french")
    return " ".join(_FR_STEMMER.stemWords(text_lexical.split()))


def _minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Per-query min-max normalize a [Nq x F] score matrix to [0, 1] (score-fusion prep) —
    the vectorized twin of `hybrid._minmax`."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / np.clip(hi - lo, 1e-9, None)


def _fusion_metrics(fused: np.ndarray, targets: np.ndarray, ks: list[int]) -> dict:
    """Overall (micro) MRR + hit@k for a [Nq x F] fused-score matrix, given each query's true
    fiche: rank the fused scores, then average 1/rank and (rank <= k) over all queries."""
    rank = _ranks(fused)[np.arange(len(targets)), targets]  # rank of the true fiche per query
    out = {"MRR": float((1.0 / rank).mean())}
    for k in ks:
        out[f"hit@{k}"] = float((rank <= k).mean())
    return out


def fusion_stemming_sweep() -> None:
    """Reproduce reports/fusion_stemming.md (dev only): RRF vs score-fusion (dense 0.3/0.5/0.7),
    each with/without BM25 stemming. Reuses the cached e5-base index (no chunk re-embedding)."""
    if not settings.embeddings_path.exists():
        raise FileNotFoundError("index not found — run `make index` first")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    dev = pd.read_parquet(settings.questions_dev_path)
    targets = dev["fiche_id"].to_numpy()
    ks = settings.eval_k

    print("encoding dev queries (once) ...", flush=True)
    query_vecs = embedder.encode([canonicalize(q) for q in dev["question"]], kind="query")
    queries_lex = [lexical_normalize(q) for q in dev["question"]]
    queries_stem = [_stem_lexical(q) for q in queries_lex]

    # Reuse the cached index: chunks.parquet (text_lexical + fiche_id) + embeddings.npy (aligned).
    chunks = pd.read_parquet(settings.chunks_path)
    chunk_fiche = chunks["fiche_id"].to_numpy()
    chunk_emb = np.load(settings.embeddings_path)
    dense = _max_pool(query_vecs @ chunk_emb.T, chunk_fiche)

    # BM25 fiche-score matrix from a corpus + matching queries (both already normalized).
    def bm25(corpus: list[str], queries: list[str]) -> np.ndarray:
        bm = BM25Okapi([t.split() for t in corpus], k1=settings.bm25_k1, b=settings.bm25_b)
        return _max_pool(np.vstack([bm.get_scores(q.split()) for q in queries]), chunk_fiche)

    bm = bm25(chunks["text_lexical"].tolist(), queries_lex)  # unstemmed (the production text)
    # Stemmed variant — stem both sides (corpus + queries) so they match.
    bm_stem = bm25([_stem_lexical(t) for t in chunks["text_lexical"]], queries_stem)

    c = settings.rrf_c
    dense_rank = _ranks(dense)  # dense side is identical across all rows — compute once
    dn = _minmax_rows(dense)  # ...and its normalized form, reused by every score-fusion row

    # The two fusion methods under comparison, as functions of a BM25 score matrix `b`:
    def rrf(b: np.ndarray) -> np.ndarray:  # rank-based (the baseline)
        return 1.0 / (c + dense_rank) + 1.0 / (c + _ranks(b))

    def score_fusion(b: np.ndarray, w: float) -> np.ndarray:  # score-based, dense weight `w`
        return w * dn + (1.0 - w) * _minmax_rows(b)

    rows = {
        "hybrid RRF (baseline)": _fusion_metrics(rrf(bm), targets, ks),
        "RRF + stemming": _fusion_metrics(rrf(bm_stem), targets, ks),
        "score-fusion dense0.3/bm250.7": _fusion_metrics(score_fusion(bm, 0.3), targets, ks),
        "score-fusion dense0.5/bm250.5": _fusion_metrics(score_fusion(bm, 0.5), targets, ks),
        "score-fusion dense0.7/bm250.3": _fusion_metrics(score_fusion(bm, 0.7), targets, ks),
        "score-fusion 0.7/0.3 + stemming": _fusion_metrics(score_fusion(bm_stem, 0.7), targets, ks),
    }
    table = pd.DataFrame(rows).T[["MRR"] + [f"hit@{k}" for k in ks]].round(3)
    report = "\n".join(
        [
            "# Confidence-aware fusion + stemming — dev set",
            "",
            "> Dev (1,004), test frozen. Reuses the cached e5-base index (no re-embedding).",
            "> **score-fusion** = per-query min-max normalize each retriever, then weighted sum"
            " (`dense_w / bm25_w`). **stemming** = Snowball French on the BM25 text.",
            "> Reproduce: `make fusion-stemming` (needs `uv sync --group ml`). Stemming is the"
            " *rejected* lever (no dev hit@1 gain), kept reproducible like the RRF baseline.",
            "",
            table.to_markdown(),
            "",
        ]
    )
    settings.reports_dir.joinpath("fusion_stemming.md").write_text(report)
    print(f"\nwrote {settings.reports_dir / 'fusion_stemming.md'}\n", flush=True)
    print(table.to_markdown())


def main() -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    fiche_cat = {f.fiche_id: _theme_label(f.category) for f in load_fiches()}
    dev = _add_cat(pd.read_parquet(settings.questions_dev_path), fiche_cat)

    # Encode the dev queries ONCE (batched) and reuse across every variant.
    print("encoding dev queries (once, batched) ...", flush=True)
    query_vecs = embedder.encode([canonicalize(q) for q in dev["question"]], kind="query")
    queries_lex = [lexical_normalize(q) for q in dev["question"]]
    targets = dev["fiche_id"].to_numpy()
    cats = dev["cat"].to_numpy()

    # The four chunking configs to compare (each re-chunks + re-embeds the corpus in memory).
    variants = {
        "baseline (hybrid)": dict(prepend_title=False, semantic=False),
        "+ title prepend": dict(prepend_title=True, semantic=False),
        "+ semantic chunk": dict(prepend_title=False, semantic=True),
        "+ both": dict(prepend_title=True, semantic=True),
    }

    results: dict[str, dict] = {}
    n_chunks: dict[str, int] = {}
    for name, opts in variants.items():
        print(f"variant: {name} ...", flush=True)
        chunks = build_variant(embedder, **opts)
        n_chunks[name] = len(chunks)
        results[name] = eval_hybrid_fast(
            chunks,
            query_vecs,
            queries_lex,
            targets,
            cats,
            embedder,
            settings.eval_k,
            settings.rrf_c,
        )

    table = _overall_table(results)
    per_theme = pd.DataFrame(
        {name: res["per_theme"]["hit@1"] for name, res in results.items()}
    ).round(3)
    counts = "\n".join(f"- **{name}**: {n} chunks" for name, n in n_chunks.items())
    report = "\n".join(
        [
            "# Hybrid experiments — dev set",
            "",
            "> Dev-only. The **test set is held out** for a single final run on the chosen",
            "> config. All rows use the **hybrid** retriever; chunk size fixed at 256.",
            "",
            "## Overall (dev)",
            "",
            table.to_markdown(),
            "",
            "## Per-theme hit@1 (dev)",
            "",
            per_theme.to_markdown(),
            "",
            "## Chunk counts",
            "",
            counts,
            "",
        ]
    )
    settings.reports_dir.joinpath("experiments.md").write_text(report)
    print(f"\nwrote {settings.reports_dir / 'experiments.md'}\n", flush=True)
    print(table.to_markdown())


if __name__ == "__main__":
    import sys

    # arg "fusion-stemming" runs the fusion sweep; a bare call runs the chunking ablations.
    if "fusion-stemming" in sys.argv[1:]:
        fusion_stemming_sweep()
    else:
        main()
