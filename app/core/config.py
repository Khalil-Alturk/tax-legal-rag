"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables live here so nothing is hardcoded in the RAG code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SecretStr keeps the key out of repr()/logs: printing settings shows '**********'.
    google_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"
    llm_temperature: float = 0.0

    data_dir: Path = Path("data")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "tax_legal_docs"

    chunk_size: int = Field(800, gt=100)
    chunk_overlap: int = Field(150, ge=0)

    default_top_k: int = Field(4, ge=1)
    max_top_k: int = Field(20, ge=1)

    embed_batch_size: int = Field(20, ge=1)
    embed_max_retries: int = Field(5, ge=1)

    max_upload_mb: int = Field(10, ge=1)
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read .env once per process)."""
    return Settings()
