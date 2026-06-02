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

    @property
    def kb_path(self) -> Path:
        """Absolute path to the knowledge-base CSV (113 fiches)."""
        return self.raw_data_dir / self.kb_csv

    @property
    def questions_path(self) -> Path:
        """Absolute path to the eval question CSV."""
        return self.raw_data_dir / self.questions_csv


settings = Settings()
