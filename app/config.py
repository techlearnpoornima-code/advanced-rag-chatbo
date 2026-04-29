"""
Configuration management for the Advanced RAG Chatbot
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    ANTHROPIC_API_KEY: str
    
    # Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    
    # Vector Database
    VECTOR_DB_PATH: Path = Path("./data/vectordb")
    VECTOR_DB_TYPE: str = "chromadb"
    
    # Chunking Strategy
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    MAX_CHUNKS_PER_DOC: int = 50
    
    # Retrieval Configuration
    TOP_K: int = 10
    RERANK_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    ENABLE_RERANKING: bool = True
    ENABLE_HYBRID_SEARCH: bool = True
    
    # Multi-Intent Configuration
    ENABLE_MULTI_INTENT: bool = True
    MAX_SUB_QUERIES: int = 3
    
    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600
    CACHE_DIR: Path = Path("./data/cache")
    
    # Data Ingestion
    STACKOVERFLOW_DATA_PATH: Path = Path("./data/raw/stackoverflow")
    MAX_DOCUMENTS: int = 50000
    MIN_SCORE: int = 5
    MIN_ANSWER_LENGTH: int = 100
    
    # Application Settings
    APP_NAME: str = "Advanced RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/app.log")
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Intent-specific configurations
INTENT_CONFIGS = {
    "factual": {
        "top_k": 3,
        "temperature": 0.3,
        "max_tokens": 500,
        "filter_tags": True,
        "description": "Concise definitions and facts"
    },
    "howto": {
        "top_k": 5,
        "temperature": 0.5,
        "max_tokens": 1500,
        "require_code": True,
        "description": "Step-by-step tutorials and guides"
    },
    "comparison": {
        "top_k": 8,
        "temperature": 0.4,
        "max_tokens": 2000,
        "dual_retrieval": True,
        "description": "Side-by-side comparisons"
    },
    "reasoning": {
        "top_k": 2,
        "temperature": 0.7,
        "max_tokens": 1500,
        "min_context": True,
        "description": "Conceptual explanations"
    },
    "code": {
        "top_k": 5,
        "temperature": 0.2,
        "max_tokens": 2000,
        "code_only": True,
        "description": "Code examples and implementations"
    },
    "debug": {
        "top_k": 6,
        "temperature": 0.3,
        "max_tokens": 1500,
        "include_errors": True,
        "description": "Error resolution and debugging"
    }
}
