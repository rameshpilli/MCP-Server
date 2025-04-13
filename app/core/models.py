from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Enum as SQLEnum, Boolean, ForeignKey, MetaData
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.config import ModelBackend
from app.core.database import Base

# Create metadata with naming convention
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

naming_metadata = MetaData(naming_convention=convention)
Base.metadata = naming_metadata

class ModelRecord(Base):
    """Model registration record."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, unique=True, index=True)
    name = Column(String)
    version = Column(String)
    backend = Column(SQLEnum(ModelBackend))
    api_base = Column(String)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    config = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    
    # Add fields for status tracking
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)  # Total tokens processed by the model
    last_used = Column(DateTime, nullable=True)
    average_latency = Column(Float, default=0.0)

    # Relationships
    usage_logs = relationship("ModelUsageLog", back_populates="model")

class APIKey(Base):
    """Database model for API keys"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    key_id = Column(String(255), unique=True, nullable=False)
    key = Column(String(255), unique=True, nullable=False)
    owner = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    permissions = Column(JSON, default=["read"])
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    rate_limit = Column(String(50), default="20/minute")
    
    # Relationships
    usage_logs = relationship("ModelUsageLog", back_populates="api_key")
    quotas = relationship("UsageQuota", backref="api_key")

class ModelUsageLog(Base):
    """Model usage log."""
    __tablename__ = "model_usage_logs"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, ForeignKey("models.model_id"))
    api_key_id = Column(Integer, ForeignKey('api_keys.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    request_type = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    latency = Column(Float)  # in seconds
    status = Column(String)
    error = Column(String, nullable=True)
    request_metadata = Column(JSON, default=dict)

    # Relationships
    model = relationship("ModelRecord", back_populates="usage_logs")
    api_key = relationship("APIKey", back_populates="usage_logs")

class UsageQuota(Base):
    """Database model for usage quotas"""
    __tablename__ = 'usage_quotas'

    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey('api_keys.id'), nullable=False)
    quota_type = Column(String(50))  # e.g., 'daily', 'monthly'
    max_requests = Column(Integer)
    max_tokens = Column(Integer)
    max_cost = Column(Float)
    reset_frequency = Column(String(50))  # e.g., 'daily', 'monthly'
    last_reset = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 