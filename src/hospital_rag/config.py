from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Connection settings, read from the .env file (or from environment variables)."""
    model_config = SettingsConfigDict(env_file=".env")

    postgres_url: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str
    embeddings_url: str = "http://localhost:8001/v1"
    rerank_url: str = "http://localhost:8002/rerank"
    llm_url: str = "http://localhost:8000/v1"


settings = Settings()