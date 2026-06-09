"""Central configuration (paths, model names, params).

Values come from the defaults below, overridable via environment variables or a `.env`
file at the project root.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "processed" / "index"

    # --- Data files ---
    kb_csv: str = "info_particulier_impot.csv"
    questions_csv: str = "questions_fiches_fip.csv"

    # --- Retrieval ---
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_revision: str = ""  # pin a HF commit for full reproducibility; "" = latest
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    query_prefix: str = "query: "  # e5 requires query/passage prefixes
    passage_prefix: str = "passage: "
    top_k: int = 5

    # --- Hybrid fusion ---
    # Score-based fusion (per-query min-max normalize, then weighted sum). Score fusion beat
    # RRF on dev; 0.5 and 0.7 tie on hit@1, so we keep the balanced 0.5/0.5 (the simpler,
    # standard choice — reproduce the sweep with `make fusion-stemming`).
    fusion_dense_weight: float = 0.5  # BM25 weight = 1 - this

    # --- Evaluation ---
    eval_k: list[int] = [1, 3, 5, 10]  # hit@k cutoffs to report
    rrf_c: int = 60  # Reciprocal Rank Fusion constant (kept for the dev experiments)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- Chunking (size measured in whitespace "words" as a token proxy) ---
    chunk_cap: float = 256.0  # max chunk size; set to math.inf for whole-fiche chunks
    chunk_overlap: int = 32
    strip_title: bool = False

    # --- Eval split ---
    test_size: float = 0.30
    random_seed: int = 42
    stratify_by: str = "fiche_id"

    @property
    def kb_path(self) -> Path:
        """Absolute path to the knowledge-base CSV (113 fiches)."""
        return self.raw_data_dir / self.kb_csv

    @property
    def questions_path(self) -> Path:
        """Absolute path to the eval question CSV."""
        return self.raw_data_dir / self.questions_csv

    # --- Output artifacts (under data/processed/) ---
    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.parquet"

    @property
    def chunks_sample_path(self) -> Path:
        return self.processed_dir / "chunks_sample.jsonl"

    @property
    def questions_dev_path(self) -> Path:
        return self.processed_dir / "questions_dev.parquet"

    @property
    def questions_test_path(self) -> Path:
        return self.processed_dir / "questions_test.parquet"

    # --- Index artifacts ---
    @property
    def embeddings_path(self) -> Path:
        return self.processed_dir / "embeddings.npy"

    @property
    def embeddings_meta_path(self) -> Path:
        return self.processed_dir / "embeddings_meta.parquet"

    @property
    def embeddings_info_path(self) -> Path:
        return self.processed_dir / "embeddings_info.json"

    @property
    def eval_report_path(self) -> Path:
        return self.reports_dir / "eval.md"


settings = Settings()
