from datetime import datetime, timedelta, UTC
from typing import Dict, Optional, Set, List
import secrets
import logging
from pydantic import BaseModel
import sqlite3
import os
from dataclasses import dataclass
import json
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .models import ModelRecord, APIKey as DBAPIKey
from .database import get_db
from .logger import logger
from .security import APIKeyManager, create_api_key_hash

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    key_id: str
    key: str
    owner: str
    created_at: datetime
    expires_at: datetime
    permissions: List[str]
    is_active: bool
    last_used: Optional[datetime]
    usage_count: int
    rate_limit: str

    @classmethod
    def from_db_row(cls, row: tuple) -> 'APIKey':
        return cls(
            key_id=row[0],
            key=row[1],
            owner=row[2],
            created_at=datetime.fromisoformat(row[3]),
            expires_at=datetime.fromisoformat(row[4]),
            permissions=json.loads(row[5]),
            is_active=bool(row[6]),
            last_used=datetime.fromisoformat(row[7]) if row[7] else None,
            usage_count=row[8],
            rate_limit=row[9]
        )

class APIKeyCreate(BaseModel):
    """API key creation model"""
    owner: str
    expires_at: Optional[datetime] = None
    permissions: List[str] = ["read"]
    rate_limit: str = "20/minute"

# Initialize the API key manager
api_key_manager = APIKeyManager()

X_API_KEY = APIKeyHeader(name="X-API-Key", auto_error=True)
TEST_API_KEY = os.getenv("TEST_API_KEY", "test_key_dev_only")
IS_DEV_MODE = os.getenv("APP_ENV", "development") == "development"

class ModelAuth:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_model_by_key(self, api_key: str) -> Optional[ModelRecord]:
        """Get model record by API key"""
        if IS_DEV_MODE and api_key == TEST_API_KEY:
            # In dev mode, return a test model with full permissions
            return ModelRecord(
                id="test_model",
                name="Test Model",
                api_key=TEST_API_KEY,
                is_active=True,
                config={
                    "allowed_sources": ["*"],
                    "permissions": ["*"],
                    "rate_limit": "1000/minute"
                }
            )
        
        return await self.db.get(ModelRecord, ModelRecord.api_key == api_key and ModelRecord.is_active == True)

    async def validate_api_key(self, api_key: str = Security(X_API_KEY)) -> ModelRecord:
        """Validate API key and return model record"""
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key is required"
            )
        
        model = await self.get_model_by_key(api_key)
        if not model:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key"
            )
        
        if not model.is_active:
            raise HTTPException(
                status_code=403,
                detail="Model is not active"
            )
        
        return model

async def get_current_model(
    api_key: str = Security(X_API_KEY),
    db: AsyncSession = Depends(get_db)
) -> ModelRecord:
    """Dependency to get current authenticated model"""
    auth = ModelAuth(db)
    return await auth.validate_api_key(api_key)

def check_model_permissions(required_permissions: list[str]):
    """Decorator to check model permissions"""
    async def check_permissions(model: ModelRecord = Depends(get_current_model)):
        if "*" in model.config.get("permissions", []):
            return model
            
        missing_permissions = [
            perm for perm in required_permissions 
            if perm not in model.config.get("permissions", [])
        ]
        
        if missing_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permissions: {', '.join(missing_permissions)}"
            )
        return model
        
    return check_permissions 

async def get_current_api_key(
    api_key: str = Security(X_API_KEY),
    db: AsyncSession = Depends(get_db)
) -> DBAPIKey:
    """Get current API key from request."""
    try:
        result = await db.execute(
            select(DBAPIKey).where(
                DBAPIKey.key == api_key,
                DBAPIKey.is_active == True,
                (DBAPIKey.expires_at.is_(None) | (DBAPIKey.expires_at > datetime.now(UTC)))
            )
        )
        api_key_record = result.scalar_one_or_none()
        if not api_key_record:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key"
            )
        
        # Update last used timestamp and usage count
        api_key_record.last_used = datetime.now(UTC)
        api_key_record.usage_count += 1
        await db.commit()
        
        return api_key_record
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        await db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        ) 