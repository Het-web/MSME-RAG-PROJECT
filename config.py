from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MSME Advisory Assistant"
    environment: str = "development"

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    router_model: str = "llama-3.1-8b-instant"
    final_provider: Literal["groq", "openrouter"] = "groq"
    final_model: str = "llama-3.3-70b-versatile"
    fallback_final_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_output_tokens: int = 1200

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:latest"

    chroma_persist_dir: Path = Path("./ragdb")
    chroma_collection_name: str = "msms_test"
    pdf_dir: Path = Path("./data/pdfs")
    ingestion_manifest_path: Path = Path("./ragdb/ingestion_manifest.json")

    retrieval_top_k: int = 8
    chunk_size: int = 1000
    chunk_overlap: int = 150

    tavily_max_results: int = 3
    tavily_search_depth: str = "advanced"
    tavily_exclude_domains: list[str] = ["msme.gov.in/sites/default/files"]

    request_timeout_seconds: int = 60
    run_ingestion_on_startup: bool = True
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
