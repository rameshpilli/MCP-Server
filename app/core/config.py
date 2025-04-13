import os
from pathlib import Path
from typing import Set, Dict, Optional, List, Union, Any
from enum import Enum
from datetime import datetime
from .database import db, ModelRecord, ModelUsageLog

class ModelBackend(str, Enum):
    """Supported model backend types"""
    CUSTOM = "custom"      # Internal custom models
    OPENAI = "openai"      # OpenAI models
    AZURE = "azure"        # Azure hosted models
    LOCAL = "local"        # Locally hosted models
    
class ModelUsageStats:
    """Track usage statistics for a model"""
    def __init__(self):
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.total_tokens: int = 0
        self.last_used: Optional[datetime] = None
        self.average_latency: float = 0.0
    
    def update(self, success: bool, tokens: int, latency: float) -> None:
        """Update usage statistics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_tokens += tokens
        self.last_used = datetime.now()
        # Update moving average of latency
        self.average_latency = (
            (self.average_latency * (self.total_requests - 1) + latency)
            / self.total_requests
        )
    
class ModelConfig:
    """Configuration for a specific model"""
    def __init__(
        self,
        model_id: str,
        backend: ModelBackend,
        api_base: str,
        api_version: str = "v1",
        timeout: int = 30,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        registration_date: Optional[datetime] = None,
        **kwargs
    ):
        self.model_id = model_id
        self.backend = backend
        self.api_base = api_base
        self.api_version = api_version
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.registration_date = registration_date or datetime.now()
        self.additional_params = kwargs
        self.usage_stats = ModelUsageStats()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert model config to dictionary for database storage"""
        return {
            "model_id": self.model_id,
            "backend": self.backend.value,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "registration_date": self.registration_date,
            "additional_params": self.additional_params,
            "usage_stats": {
                "total_requests": self.usage_stats.total_requests,
                "successful_requests": self.usage_stats.successful_requests,
                "failed_requests": self.usage_stats.failed_requests,
                "total_tokens": self.usage_stats.total_tokens,
                "last_used": self.usage_stats.last_used,
                "average_latency": self.usage_stats.average_latency
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create model config from dictionary (database record)"""
        model = cls(
            model_id=data["model_id"],
            backend=ModelBackend(data["backend"]),
            api_base=data["api_base"],
            api_version=data["api_version"],
            timeout=data["timeout"],
            max_tokens=data["max_tokens"],
            temperature=data["temperature"],
            registration_date=data["registration_date"],
            **data.get("additional_params", {})
        )
        # Restore usage stats
        stats = data.get("usage_stats", {})
        model.usage_stats.total_requests = stats.get("total_requests", 0)
        model.usage_stats.successful_requests = stats.get("successful_requests", 0)
        model.usage_stats.failed_requests = stats.get("failed_requests", 0)
        model.usage_stats.total_tokens = stats.get("total_tokens", 0)
        model.usage_stats.last_used = stats.get("last_used")
        model.usage_stats.average_latency = stats.get("average_latency", 0.0)
        return model

class ServerConfig:
    """Server configuration settings."""
    
    # Base configuration
    BASE_DIR: Path = Path(__file__).parent.parent.parent / "storage"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ITEMS_PER_PAGE: int = 50
    UPLOAD_CHUNK_SIZE: int = 1024 * 1024  # 1MB

    # File extensions
    ALLOWED_EXTENSIONS: Set[str] = {
        # Documents
        'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
        # Images
        'png', 'jpg', 'jpeg', 'gif', 'svg',
        # Archives
        'zip', 'tar', 'gz',
        # Code
        'py', 'js', 'html', 'css', 'json', 'xml', 'yaml', 'yml'
    }

    # Security settings
    API_KEYS: Set[str] = set(os.getenv('API_KEYS', 'test_key,dev_key').split(','))
    RATE_LIMIT: str = os.getenv('RATE_LIMIT', '20/minute')

    # Cache settings
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
    CACHE_MAX_SIZE: int = int(os.getenv('CACHE_MAX_SIZE', '100'))  # 100 items

    # Server settings
    PORT: int = int(os.getenv('PORT', '8000'))
    HOST: str = os.getenv('HOST', '0.0.0.0')
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'

    # Storage settings
    STORAGE_TYPE: str = os.getenv('STORAGE_TYPE', 'local')
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/mcp')
    
    # Cloud storage (if applicable)
    AZURE_STORAGE_CONNECTION: str = os.getenv('AZURE_STORAGE_CONNECTION')
    S3_BUCKET: str = os.getenv('S3_BUCKET')
    
    # Snowflake settings (if enabled)
    SNOWFLAKE_ENABLED: bool = os.getenv('SNOWFLAKE_ENABLED', 'false').lower() == 'true'
    SNOWFLAKE_ACCOUNT: str = os.getenv('SNOWFLAKE_ACCOUNT')
    SNOWFLAKE_WAREHOUSE: str = os.getenv('SNOWFLAKE_WAREHOUSE')
    SNOWFLAKE_DATABASE: str = os.getenv('SNOWFLAKE_DATABASE')
    SNOWFLAKE_SCHEMA: str = os.getenv('SNOWFLAKE_SCHEMA')
    SNOWFLAKE_ROLE: str = os.getenv('SNOWFLAKE_ROLE')
    SNOWFLAKE_USER: str = os.getenv('SNOWFLAKE_USER')

    # Monitoring settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_METRICS: bool = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    METRICS_PORT: int = int(os.getenv('METRICS_PORT', '9090'))

    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings."""
        if cls.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        
        if cls.ITEMS_PER_PAGE <= 0:
            raise ValueError("ITEMS_PER_PAGE must be positive")
            
        if cls.UPLOAD_CHUNK_SIZE <= 0:
            raise ValueError("UPLOAD_CHUNK_SIZE must be positive")
            
        if not cls.ALLOWED_EXTENSIONS:
            raise ValueError("ALLOWED_EXTENSIONS cannot be empty")
            
        # Create storage directory if it doesn't exist
        os.makedirs(cls.BASE_DIR, exist_ok=True)

    @classmethod
    def is_file_allowed(cls, filename: str) -> bool:
        """Check if a file is allowed based on its extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

# Validate configuration on module import
ServerConfig.validate_config()

async def initialize_database():
    """Initialize the database"""
    await db.init_db()

# Note: Database initialization should be called during application startup
# Example: await initialize_database() 