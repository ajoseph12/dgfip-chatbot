"""Thin wrapper around the sentence-transformers embedding model.

Handles the two e5 requirements: the `query:` / `passage:` prefixes and L2-normalization
(so a plain dot product equals cosine similarity).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from dgfip_chatbot.config import settings


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        revision: str | None = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(
            self.model_name,
            device=device or settings.embedding_device,
            revision=(revision or settings.embedding_revision) or None,
        )

    def encode(
        self, texts: list[str], kind: str, batch_size: int | None = None, progress: bool = False
    ) -> np.ndarray:
        """Encode `texts` as either queries or passages → L2-normalized vectors `[n, dim]`.

        `progress=True` shows a progress bar (useful for the one-time index build, which is
        otherwise silent for minutes on CPU)."""
        if kind not in ("query", "passage"):
            raise ValueError(f"kind must be 'query' or 'passage', got {kind!r}")
        prefix = settings.query_prefix if kind == "query" else settings.passage_prefix
        prefixed = [prefix + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size or settings.embedding_batch_size,
            normalize_embeddings=True,  # L2-normalize → dot product == cosine
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
