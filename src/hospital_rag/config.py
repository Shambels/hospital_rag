from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Machine settings, read from the .env file (or from environment variables)."""
    # extra="ignore": .env also holds DB_USER, DB_PASSWORD and DB_NAME, which only docker-compose reads
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_url: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str
    embeddings_url: str = "http://localhost:8001/v1"
    rerank_url: str = "http://localhost:8002/rerank"
    llm_url: str = "http://localhost:8000/v1"
    judge_url: str = "http://localhost:8003/v1"

    # Relative to the repo root: scripts are run from there (uv run python scripts/...)
    corpus_path: Path = Path("data/docs")
    documents_path: Path = Path("data/hopital_documents.json")
    eval_path: Path = Path("data/hopital_eval.json")
    conversions_path: Path = Path("data/conversions")
    models_path: Path  # holds embedder/{name}, reranker/{name} and docling/

    @property
    def docling_models_path(self):
        return self.models_path / "docling"


class OcrSettings(BaseSettings):
    """Device and threads of the Docling PDF models (layout, tables, OCR), read from OCR_DEVICE, OCR_NUM_THREADS."""
    ocr_device: str = "cpu"
    ocr_num_threads: int = 10


settings = Settings()
ocr_settings = OcrSettings()
