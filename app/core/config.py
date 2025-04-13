from enum import Enum
from typing import Optional, List, Set, ClassVar
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from functools import lru_cache

class ModelBackend(str, Enum):
    """Supported model backends."""
    LOCAL = "local"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class StorageBackend(str, Enum):
    """Storage backend types"""
    LOCAL = "local"
    AZURE = "azure"

class Settings(BaseSettings):
    """Application settings."""
    
    # Testing Configuration
    TESTING: bool = False
    TEST_DB_URL: str = "sqlite+aiosqlite:///:memory:"
    
    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    BASE_DIR: str = os.path.join(os.getcwd(), 'data')
    
    # Database Configuration
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "mcp"
    
    # Azure SQL Database settings
    AZURE_SQL_CONNECTION_STRING: Optional[str] = None
    
    # Security
    SECRET_KEY: str = Field(default="dev_secret_key", description="Secret key for JWT encoding")
    ALGORITHM: str = "HS256"
    
    # API Key settings
    API_KEY_LENGTH: int = 32
    API_KEY_PREFIX: str = "mcp"
    DEFAULT_API_KEY_EXPIRY_DAYS: int = 30
    API_KEY_EXPIRY_DAYS: int = Field(default=365, description="Number of days until API keys expire")
    DEFAULT_RATE_LIMIT: str = "20/minute"
    TEST_API_KEY: str = "test_key_dev_only"
    API_KEYS: Set[str] = {"test_key", "dev_key"}
    DEFAULT_PERMISSIONS: List[str] = ["read", "write", "execute"]
    
    # Rate Limiting
    RATE_LIMIT_CLEANUP_INTERVAL: int = 3600  # 1 hour in seconds
    RATE_LIMIT_DEFAULT_LIMIT: int = 20
    RATE_LIMIT_DEFAULT_WINDOW: str = "minute"
    
    # Cache
    CACHE_TTL: int = 300  # 5 minutes
    
    # File Storage
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_CHUNK_SIZE: int = 1048576  # 1MB
    STORAGE_PATH: str = "storage"
    ALLOWED_EXTENSIONS: Set[str] = {".txt", ".log", ".py", ".json", ".yaml", ".yml", ".md", ".csv"}
    
    # Storage Backend
    STORAGE_BACKEND: StorageBackend = StorageBackend.LOCAL
    
    # Azure Blob Storage
    AZURE_STORAGE_ACCOUNT: Optional[str] = None
    AZURE_CONTAINER_NAME: str = "mcp-logs"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/mcp_server.log"
    
    # Model Configuration
    DEFAULT_MODEL_TIMEOUT: int = 30
    MODEL_REGISTRY_PATH: str = "data/model_registry.json"
    
    # External API Keys
    WEATHER_API_KEY: str = ""
    NEWS_API_KEY: str = ""

    # Public paths that don't require authentication
    PUBLIC_PATHS: Set[str] = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/health",
        "/metrics",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/keys",
        "/api/v1/auth/api-keys",
    }
    
    # Paths that require authentication but not rate limiting
    NO_RATE_LIMIT_PATHS: Set[str] = {
        "/api/models/register",
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"

    def get_db_url(self) -> str:
        """Get the database URL based on the storage backend."""
        if self.TESTING:
            return self.TEST_DB_URL
        if self.STORAGE_BACKEND == StorageBackend.AZURE:
            return self.AZURE_SQL_CONNECTION_STRING or ""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
