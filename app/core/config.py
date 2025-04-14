"""Application configuration."""

import os
from enum import Enum
from typing import Optional, Set, Dict, Any
from pydantic_settings import BaseSettings
from pathlib import Path

# Import MCP configuration classes
from mcp.database.config import DatabaseConfig
from mcp.server.config import ServerConfig
from mcp.storage.config import StorageConfig

class ModelBackend(str, Enum):
    """Types of model backends supported."""
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    LOCAL = "local"

class StorageBackend(str, Enum):
    """Types of storage backends supported."""
    LOCAL = "local"
    AZURE = "azure"
    S3 = "s3"

class Settings(BaseSettings):
    """Application settings."""
    # Application
    APP_NAME: str = "MCP Server"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    TESTING: bool = os.getenv("TESTING", "False").lower() == "true"
    
    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_NAME: str = os.getenv("DB_NAME", "mcp")
    DB_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # API configuration
    API_PREFIX: str = "/api/v1"
    
    # API key settings
    SECRET_KEY: str
    API_KEY_EXPIRY_DAYS: int = 30
    ALGORITHM: str = "HS256"
    
    # Rate limiting
    RATE_LIMIT: str = "20/minute"
    RATE_LIMIT_WINDOW: int = 60  # Window in seconds
    
    # Caching
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
        "/",  # Landing page
        "/docs",  # Swagger UI
        "/redoc",  # ReDoc UI
        "/openapi.json",  # OpenAPI schema
        "/health"  # Health check endpoint
    }
    
    # Paths that require authentication but not rate limiting
    NO_RATE_LIMIT_PATHS: Set[str] = {
        "/health",  # Health check endpoint
        "/docs",  # Swagger UI
        "/redoc",  # ReDoc UI
        "/openapi.json"  # OpenAPI schema
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"

    def get_db_url(self) -> str:
        """Get database URL from environment or construct from components."""
        if self.DB_URL:
            return self.DB_URL
        
        # Construct SQLAlchemy URL
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def get_mcp_database_config(self) -> DatabaseConfig:
        """Get MCP DatabaseConfig using environment variables"""
        return DatabaseConfig(
            url=self.get_db_url(),
            echo=self.DEBUG
            # MCP will read these from environment variables
            # or we can pass them explicitly if needed
        )
    
    def get_mcp_server_config(self) -> ServerConfig:
        """Get MCP ServerConfig using environment variables"""
        return ServerConfig(
            name=self.APP_NAME,
            version="1.0.0",
            # Other server configuration options
        )
    
    def get_mcp_storage_config(self) -> StorageConfig:
        """Get MCP StorageConfig using environment variables"""
        return StorageConfig(
            # Storage configuration options
            type=self.STORAGE_BACKEND.value,
            # MCP will handle mapping the rest from env vars
        )

# Singleton pattern for settings
_settings = None

def get_settings() -> Settings:
    """Get settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
