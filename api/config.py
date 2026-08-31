import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:8b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME")

    TOP_K: int = int(os.getenv("TOP_K", "5"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Benchmark-only configuration
    BENCHMARK_PROVIDER: str = os.getenv("BENCHMARK_PROVIDER", "openai")
    BENCHMARK_MODEL: str = os.getenv("BENCHMARK_MODEL", "...")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "...")


settings = Settings()
