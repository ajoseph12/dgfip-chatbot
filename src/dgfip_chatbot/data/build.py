"""Phase 1 pipeline: raw CSVs -> chunks.parquet + questions_{dev,test}.parquet.

Run with ``make data`` (or ``uv run python -m dgfip_chatbot.data.build``).
"""

import pandas as pd

from dgfip_chatbot.config import settings
from dgfip_chatbot.data.chunk import chunk_text, count_words
from dgfip_chatbot.data.clean import canonicalize, lexical_normalize
from dgfip_chatbot.data.loaders import load_fiches, load_questions
from dgfip_chatbot.data.schema import Chunk
from dgfip_chatbot.data.split import stratified_split


def build_chunks() -> pd.DataFrame:
    """Clean + chunk every fiche into a flat chunk table (deterministic)."""
    rows: list[dict] = []
    for fiche in load_fiches():
        # 1) clean the full fiche text (keeps \n so the chunker can cut on it)
        clean = canonicalize(fiche.texte, strip_title=settings.strip_title, title=fiche.titre)
        # 2) split into chunks, then build one table row per chunk
        for i, piece in enumerate(
            chunk_text(clean, cap=settings.chunk_cap, overlap=settings.chunk_overlap)
        ):
            rows.append(
                Chunk(
                    chunk_id=f"{fiche.fiche_id}-{i}",  # unique: parent fiche + position
                    fiche_id=fiche.fiche_id,  # link back to the fiche (the graded unit)
                    chunk_index=i,
                    text=piece,  # natural text → embeddings + display
                    text_lexical=lexical_normalize(piece),  # normalized text → BM25
                    titre=fiche.titre,
                    url=fiche.url,
                    category=fiche.category,
                    subcategory=fiche.subcategory,
                    n_chars=len(piece),
                    n_tokens=count_words(piece),
                ).model_dump()  # pydantic model -> plain dict for the DataFrame
            )
    return pd.DataFrame(rows)


def main() -> None:
    # data/processed/ is gitignored; create it on demand.
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    # --- knowledge base: cleaned + chunked table (+ a small JSONL sample to eyeball) ---
    chunks = build_chunks()
    chunks.to_parquet(settings.chunks_path, index=False)
    chunks.head(20).to_json(
        settings.chunks_sample_path, orient="records", lines=True, force_ascii=False
    )

    # --- eval questions: stratified 70/30 dev/test split ---
    dev, test = stratified_split(
        load_questions(),
        test_size=settings.test_size,
        seed=settings.random_seed,
        stratify_by=settings.stratify_by,
    )
    dev.to_parquet(settings.questions_dev_path, index=False)
    test.to_parquet(settings.questions_test_path, index=False)

    print(
        f"chunks: {len(chunks)} from {chunks['fiche_id'].nunique()}/113 fiches | "
        f"dev: {len(dev)} | test: {len(test)} -> {settings.processed_dir}"
    )


if __name__ == "__main__":
    main()
