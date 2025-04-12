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
    """Server configuration settings.
    
    This class centralizes all server configuration settings including:
    - File storage settings (directory, size limits, allowed types)
    - Pagination settings
    - Upload configurations
    - API and security settings
    - Server runtime configurations
    - LLM Integration settings
    
    Flow:
    1. UI/Client sends request to MCP
    2. MCP looks up appropriate model from registered models
    3. MCP forwards request to model's API endpoint
    4. Response is returned to client
    5. Usage statistics are updated and logged
    
    Example:
    ```python
    from app.core.config import ServerConfig, ModelConfig, ModelBackend
    
    # Register custom model
    custom_model = ModelConfig(
        model_id="custom-model-v1",
        backend=ModelBackend.CUSTOM,
        api_base="http://internal-llm-cluster:8000",
        api_version="v2",
        timeout=60
    )
    
    # Add model to configuration and database
    await ServerConfig.register_model(custom_model)
    
    # Use model in your application
    async with ServerConfig.get_model("custom-model-v1") as model:
        response = await model.generate(prompt="Your prompt here")
        # Usage stats are automatically updated and logged
    ```
    """
    
    # Base directory for file operations
    BASE_DIR: Path = Path(__file__).parent.parent.parent / "storage"
    
    # Maximum file size (100 MB)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS: Set[str] = {
        # Documents
        'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
        # Images
        'png', 'jpg', 'jpeg', 'gif', 'svg',
        # Archives
        'zip', 'tar', 'gz',
        # Code
        'py', 'js', 'html', 'css', 'json', 'xml',
    }
    
    # Maximum items per page for directory listing
    ITEMS_PER_PAGE: int = 50
    
    # Chunk size for file uploads (1 MB)
    UPLOAD_CHUNK_SIZE: int = 1024 * 1024

    # Security and API settings
    API_KEYS = {"test_key", "dev_key"}  # Sample API keys
    CACHE_TTL = 300  # Cache TTL in seconds (5 minutes)
    RATE_LIMIT = "20/minute"  # Rate limit per IP
    
    # Server settings
    PORT = int(os.getenv('PORT', 8000))
    HOST = os.getenv('HOST', '0.0.0.0')

    # LLM Integration Settings
    _registered_models: Dict[str, ModelConfig] = {}
    DEFAULT_MODEL_ID: str = os.getenv('DEFAULT_MODEL_ID', 'default-model')
    LLM_REQUEST_TIMEOUT: int = int(os.getenv('LLM_REQUEST_TIMEOUT', '30'))
    LLM_RATE_LIMIT: str = os.getenv('LLM_RATE_LIMIT', '60/minute')
    LLM_CONCURRENT_REQUESTS: int = int(os.getenv('LLM_CONCURRENT_REQUESTS', '5'))
    
    @classmethod
    async def register_model(cls, model_config: ModelConfig) -> None:
        """Register a new model configuration and save to database.
        
        Args:
            model_config: Configuration for the model to register
            
        Raises:
            ValueError: If a model with the same ID is already registered
        """
        if model_config.model_id in cls._registered_models:
            raise ValueError(f"Model {model_config.model_id} is already registered")
        
        # Create database record
        model_record = ModelRecord(
            **model_config.to_dict()
        )
        
        # Save to database
        async with db.get_session() as session:
            session.add(model_record)
            await session.commit()
        
        # Add to in-memory registry
        cls._registered_models[model_config.model_id] = model_config
        
        # Log registration to Azure if configured
        await db.log_to_azure(
            model_config.to_dict(),
            "model_registrations"
        )
    
    @classmethod
    async def update_model_stats(
        cls,
        model_id: str,
        success: bool,
        tokens: int,
        latency: float,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update model usage statistics and log to database.
        
        Args:
            model_id: ID of the model to update
            success: Whether the request was successful
            tokens: Number of tokens used
            latency: Request latency in seconds
            error_message: Optional error message if request failed
            metadata: Optional request metadata
        """
        # Update in-memory stats
        model = cls._registered_models.get(model_id)
        if model:
            model.usage_stats.update(success, tokens, latency)
        
        # Create usage log record
        usage_log = ModelUsageLog(
            model_id=model_id,
            success=success,
            tokens_used=tokens,
            latency=latency,
            error_message=error_message,
            request_metadata=metadata
        )
        
        # Save to database
        async with db.get_session() as session:
            # Update model record
            model_record = await session.get(ModelRecord, model_id)
            if model_record:
                model_record.total_requests += 1
                model_record.successful_requests += int(success)
                model_record.failed_requests += int(not success)
                model_record.total_tokens += tokens
                model_record.last_used = datetime.utcnow()
                model_record.average_latency = (
                    (model_record.average_latency * (model_record.total_requests - 1) + latency)
                    / model_record.total_requests
                )
            
            # Add usage log
            session.add(usage_log)
            await session.commit()
        
        # Log to Azure if configured
        await db.log_to_azure(
            {
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat(),
                "success": success,
                "tokens": tokens,
                "latency": latency,
                "error_message": error_message,
                "metadata": metadata
            },
            "model_usage"
        )
    
    @classmethod
    async def get_model_stats(cls, model_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for models.
        
        Args:
            model_id: Optional model ID to get stats for. If None, returns stats for all models.
            
        Returns:
            Dictionary of model statistics
        """
        async with db.get_session() as session:
            query = session.query(ModelRecord)
            if model_id:
                query = query.filter(ModelRecord.model_id == model_id)
            
            records = await query.all()
            return {
                record.model_id: {
                    "total_requests": record.total_requests,
                    "successful_requests": record.successful_requests,
                    "failed_requests": record.failed_requests,
                    "total_tokens": record.total_tokens,
                    "last_used": record.last_used,
                    "average_latency": record.average_latency
                }
                for record in records
            }

    @classmethod
    def get_model(cls, model_id: Optional[str] = None) -> ModelConfig:
        """Get configuration for a specific model.
        
        Args:
            model_id: ID of the model to retrieve. If None, returns the default model.
            
        Returns:
            ModelConfig for the requested model
            
        Raises:
            KeyError: If the requested model is not registered
        """
        model_id = model_id or cls.DEFAULT_MODEL_ID
        if model_id not in cls._registered_models:
            raise KeyError(f"Model {model_id} is not registered")
        return cls._registered_models[model_id]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Get a list of all registered model IDs."""
        return list(cls._registered_models.keys())

    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings and ensure required directories exist.
        
        Raises:
            ValueError: If any configuration values are invalid
            OSError: If storage directory cannot be created or is not writable
        """
        if cls.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        
        if cls.ITEMS_PER_PAGE <= 0:
            raise ValueError("ITEMS_PER_PAGE must be positive")
            
        if cls.UPLOAD_CHUNK_SIZE <= 0:
            raise ValueError("UPLOAD_CHUNK_SIZE must be positive")
            
        if not cls.ALLOWED_EXTENSIONS:
            raise ValueError("ALLOWED_EXTENSIONS cannot be empty")
            
        if cls.LLM_REQUEST_TIMEOUT <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT must be positive")
            
        if cls.LLM_CONCURRENT_REQUESTS <= 0:
            raise ValueError("LLM_CONCURRENT_REQUESTS must be positive")
            
        # Ensure storage directory exists and is writable
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        
        # Test directory is writable
        test_file = cls.BASE_DIR / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            raise OSError(f"Storage directory {cls.BASE_DIR} is not writable: {e}")

    @classmethod
    def is_file_allowed(cls, filename: str) -> bool:
        """Check if a file is allowed based on its extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

async def initialize_database():
    """Initialize the database"""
    await db.init_db()

# Validate configuration on module import
ServerConfig.validate_config()

# Note: Database initialization should be called during application startup
# Example: await initialize_database() 