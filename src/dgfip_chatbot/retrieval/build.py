"""Index build: embed every chunk's `text` and cache the matrix + metadata.

Run with ``make index`` (or ``uv run python -m dgfip_chatbot.retrieval.build``).
Requires the ml group (``uv sync --group ml``) and the chunks from ``make data``.
"""

import json

import numpy as np
import pandas as pd

from dgfip_chatbot.config import settings
from dgfip_chatbot.retrieval.embedder import Embedder


def main() -> None:
    if not settings.chunks_path.exists():
        raise FileNotFoundError("chunks not found — run `make data` first")
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_parquet(settings.chunks_path)
    embedder = Embedder()
    # Embed the natural `text` column (as passages). Vectors come back L2-normalized.
    print(
        f"embedding {len(chunks)} chunks on {settings.embedding_device} with {embedder.model_name} "
        "(first run also downloads the model; a few minutes on CPU)…",
        flush=True,
    )
    embeddings = embedder.encode(chunks["text"].tolist(), kind="passage", progress=True)

    np.save(settings.embeddings_path, embeddings)
    # Aligned metadata (same row order) so the retriever needs nothing else at query time.
    chunks[["chunk_id", "fiche_id", "titre", "url", "text"]].to_parquet(
        settings.embeddings_meta_path, index=False
    )
    settings.embeddings_info_path.write_text(
        json.dumps(
            {
                "model": embedder.model_name,
                "dim": int(embeddings.shape[1]),
                "n_chunks": int(embeddings.shape[0]),
            },
            indent=2,
        )
    )

    print(
        f"indexed {embeddings.shape[0]} chunks x {embeddings.shape[1]} dims "
        f"with {embedder.model_name} -> {settings.processed_dir}"
    )


if __name__ == "__main__":
    main()
