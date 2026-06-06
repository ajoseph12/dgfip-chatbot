"""`make eval`: compare dense / BM25 / hybrid on dev + test, run the chunking ablation,
and write reports/eval.md.

Run with ``make eval`` (or ``uv run python -m dgfip_chatbot.eval.run``). Needs the ml group
(``uv sync --group ml``), the chunks (``make data``), and the index (``make index``).
"""

import pandas as pd

from dgfip_chatbot.config import settings
from dgfip_chatbot.data.clean import canonicalize
from dgfip_chatbot.data.loaders import load_fiches
from dgfip_chatbot.eval.harness import evaluate
from dgfip_chatbot.retrieval.bm25 import BM25Retriever
from dgfip_chatbot.retrieval.dense import DenseRetriever
from dgfip_chatbot.retrieval.embedder import Embedder
from dgfip_chatbot.retrieval.hybrid import HybridRetriever


def _theme_label(category_url: str) -> str:
    return category_url.rstrip("/").split("/")[-1]


def _add_cat(questions: pd.DataFrame, fiche_cat: dict[int, str]) -> pd.DataFrame:
    return questions.assign(cat=questions["fiche_id"].map(fiche_cat))


def _build_whole_fiche_dense(embedder: Embedder) -> DenseRetriever:
    """In-memory dense index with ONE vector per whole fiche (the ablation baseline)."""
    fiches = load_fiches()
    texts = [canonicalize(f.texte, strip_title=settings.strip_title, title=f.titre) for f in fiches]
    embeddings = embedder.encode(texts, kind="passage")
    meta = pd.DataFrame(
        {
            "fiche_id": [f.fiche_id for f in fiches],
            "titre": [f.titre for f in fiches],
            "url": [f.url for f in fiches],
            "text": texts,
        }
    )
    return DenseRetriever(embedder=embedder, embeddings=embeddings, meta=meta)


def _overall_table(results: dict[str, dict]) -> pd.DataFrame:
    """name -> evaluate(...) dict → a (retriever × metric) table."""
    return pd.DataFrame({name: res["overall"] for name, res in results.items()}).T.round(3)


def main() -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    fiche_cat = {f.fiche_id: _theme_label(f.category) for f in load_fiches()}
    dev = _add_cat(pd.read_parquet(settings.questions_dev_path), fiche_cat)
    test = _add_cat(pd.read_parquet(settings.questions_test_path), fiche_cat)

    dense = DenseRetriever()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense=dense, bm25=bm25)
    retrievers = {"dense": dense, "BM25": bm25, "hybrid": hybrid}

    print("evaluating on dev ...")
    dev_results = {name: evaluate(r, dev) for name, r in retrievers.items()}
    print("evaluating on test ...")
    test_results = {name: evaluate(r, test) for name, r in retrievers.items()}

    print("ablation: whole-fiche vs chunked (dev) ...")
    whole = _build_whole_fiche_dense(dense.embedder)
    ablation = {
        "dense — chunked": dev_results["dense"],
        "dense — whole-fiche": evaluate(whole, dev),
    }

    per_theme_hit1 = pd.DataFrame(
        {name: res["per_theme"]["hit@1"] for name, res in test_results.items()}
    ).round(3)

    report = "\n".join(
        [
            "# Retrieval evaluation",
            "",
            "> **Caveat:** the eval questions are LLM-generated *from* the fiches, so the",
            "> absolute numbers are optimistic vs. real user phrasing. The *relative*",
            "> comparison (dense vs BM25 vs hybrid, and per-theme gaps) is the trustworthy signal.",
            "",
            "## Overall — test set",
            "",
            _overall_table(test_results).to_markdown(),
            "",
            "## Overall — dev set (tuning reference)",
            "",
            _overall_table(dev_results).to_markdown(),
            "",
            "## Per-theme hit@1 — test set",
            "",
            per_theme_hit1.to_markdown(),
            "",
            "## Ablation — whole-fiche vs chunked (dense, dev)",
            "",
            _overall_table(ablation).to_markdown(),
            "",
        ]
    )
    settings.eval_report_path.write_text(report)
    print(f"\nwrote {settings.eval_report_path}\n")
    print(_overall_table(test_results).to_markdown())


if __name__ == "__main__":
    main()
