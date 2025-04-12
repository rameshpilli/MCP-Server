from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Enum as SQLEnum
import os
from enum import Enum
from azure.storage.blob.aio import BlobServiceClient
from azure.identity.aio import DefaultAzureCredential

# Base class for SQLAlchemy models
Base = declarative_base()

class StorageBackend(str, Enum):
    """Storage backend types"""
    LOCAL = "local"
    AZURE = "azure"

class DatabaseConfig:
    """Database configuration with support for local and Azure databases"""
    
    # Local SQLite database
    LOCAL_DB_URL = "sqlite+aiosqlite:///./mcp.db"
    
    # Azure SQL Database (to be configured via environment variables)
    AZURE_DB_URL = os.getenv(
        "AZURE_DB_URL",
        "mssql+aioodbc://username:password@server.database.windows.net/mcp?driver=ODBC+Driver+17+for+SQL+Server"
    )
    
    # Azure Blob Storage
    AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
    AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "mcp-logs")
    
    # Current storage backend
    STORAGE_BACKEND = StorageBackend(os.getenv("STORAGE_BACKEND", "local"))
    
    @classmethod
    def get_db_url(cls) -> str:
        """Get the appropriate database URL based on configuration"""
        return cls.AZURE_DB_URL if cls.STORAGE_BACKEND == StorageBackend.AZURE else cls.LOCAL_DB_URL

class ModelRecord(Base):
    """Database model for storing LLM model configurations"""
    __tablename__ = "models"
    
    model_id = Column(String(100), primary_key=True)
    backend = Column(String(50), nullable=False)
    api_base = Column(String(200), nullable=False)
    api_version = Column(String(20))
    timeout = Column(Integer)
    max_tokens = Column(Integer)
    temperature = Column(Float)
    registration_date = Column(DateTime, default=datetime.utcnow)
    additional_params = Column(JSON)
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    last_used = Column(DateTime)
    average_latency = Column(Float, default=0.0)

class ModelUsageLog(Base):
    """Database model for detailed model usage logging"""
    __tablename__ = "model_usage_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(bool, nullable=False)
    tokens_used = Column(Integer)
    latency = Column(Float)
    error_message = Column(String(500))
    request_metadata = Column(JSON)

class Database:
    """Database connection manager"""
    def __init__(self):
        self.engine = create_async_engine(
            DatabaseConfig.get_db_url(),
            echo=True
        )
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self._blob_client: Optional[BlobServiceClient] = None
    
    async def init_db(self):
        """Initialize database and create tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_session(self) -> AsyncSession:
        """Get a database session"""
        return self.async_session()
    
    async def get_blob_client(self) -> Optional[BlobServiceClient]:
        """Get Azure Blob Storage client if configured"""
        if (
            DatabaseConfig.STORAGE_BACKEND == StorageBackend.AZURE
            and DatabaseConfig.AZURE_STORAGE_ACCOUNT
        ):
            if not self._blob_client:
                credential = DefaultAzureCredential()
                account_url = f"https://{DatabaseConfig.AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
                self._blob_client = BlobServiceClient(account_url, credential=credential)
            return self._blob_client
        return None
    
    async def log_to_azure(self, log_data: Dict[str, Any], log_type: str):
        """Log data to Azure Blob Storage"""
        if blob_client := await self.get_blob_client():
            container_client = blob_client.get_container_client(DatabaseConfig.AZURE_CONTAINER_NAME)
            
            # Create container if it doesn't exist
            if not await container_client.exists():
                await container_client.create_container()
            
            # Create log file name with timestamp
            timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H/%M_%S")
            blob_name = f"{log_type}/{timestamp}.json"
            
            # Upload log data
            blob_client = container_client.get_blob_client(blob_name)
            await blob_client.upload_blob(str(log_data))

# Global database instance
db = Database() 