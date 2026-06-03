"""Central configuration (paths, model names, params, secrets).

Values come from defaults below, overridable via environment variables or a `.env`
file at the project root. Later phases fill in the model/retrieval/LLM fields.
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

    # --- Retrieval (filled in Phase 2) ---
    embedding_model: str = ""
    top_k: int = 5

    # --- LLM (filled in Phase 4) ---
    llm_provider: str = "mistral"
    llm_model: str = "mistral-small-latest"
    mistral_api_key: str = ""

    # --- Chunking (Phase 1; size measured in whitespace "words" as a token proxy) ---
    chunk_cap: float = 256.0  # max chunk size; set to math.inf for whole-fiche chunks
    chunk_overlap: int = 32
    strip_title: bool = False

    # --- Eval split (Phase 1) ---
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

    # --- Phase 1 output artifacts (under data/processed/) ---
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


settings = Settings()
