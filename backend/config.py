"""
AutoFabric — Central Configuration
All settings loaded from environment variables via Pydantic BaseSettings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    groq_model_reasoning: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"

    # LangSmith Observability
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "autofabric"

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # RAG
    faiss_index_path: str = "data/faiss_index/"
    sample_docs_path: str = "data/sample_docs/"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_default: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50

    # App
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
