from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    database_url: str = "sqlite:///./tetraknowledge.db"
    embedding_provider: str = "hash"
    embedding_model: str = "hash-384"
    embedding_dimensions: int = 384
    # Use sentence_transformer with BGE/E5 or another multilingual model and re-index documents.
    llm_provider: str = "rule_based"
    llm_model: str = "demo-rule-based"
    top_k: int = 5
    max_document_bytes: int = 10_000_000
    cors_origins: str = "http://localhost:8501"
    azure_openai_endpoint: str | None = None
    azure_openai_base_url: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "tetraknowledge"
    otel_exporter_otlp_endpoint: str | None = None

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
