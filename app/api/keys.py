from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, UTC
from typing import List, Optional
from pydantic import BaseModel
import uuid

from app.core.database import get_db
from app.core.models import APIKey
from app.core.auth import api_key_manager
from app.core.middleware import get_api_key_from_header
from app.core.security import APIKeyManager
from app.core.config import Settings, get_settings
from app.schemas.api_key import APIKeyCreate, APIKeyResponse

router = APIRouter()  # Remove the prefix here since it's added in main.py
api_key_manager = APIKeyManager()

class APIKeyCreate(BaseModel):
    owner: str
    expiry_days: Optional[int] = None
    permissions: List[str] = []
    rate_limit: str = "5/minute"  # Default to 5/minute as per test expectation

class APIKeyResponse(BaseModel):
    key: str
    owner: str
    expires_at: datetime | None
    permissions: List[str]
    rate_limit: Optional[str]

class APIKeyInfo(BaseModel):
    """Response model for API key information"""
    owner: str
    expires_at: Optional[datetime]
    permissions: List[str]
    is_active: bool
    last_used: Optional[datetime]
    usage_count: int
    rate_limit: Optional[str]

@router.post("/keys", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    db: AsyncSession = Depends(get_db)
) -> APIKeyResponse:
    """Create a new API key"""
    settings = get_settings()
    expiry_days = key_data.expiry_days or settings.API_KEY_EXPIRY_DAYS
    expires_at = datetime.now(UTC) + timedelta(days=expiry_days)
    
    # Generate a unique key_id
    key_id = f"key_{uuid.uuid4().hex[:8]}"
    
    plain_key, api_key = await api_key_manager.generate_key(
        key_id=key_id,  # Pass the key_id to generate_key
        owner=key_data.owner,
        expires_at=expires_at,
        permissions=key_data.permissions,
        rate_limit=key_data.rate_limit
    )
    
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    
    return APIKeyResponse(
        key=plain_key,  # Return the plain key, not the hashed one
        owner=api_key.owner,
        expires_at=api_key.expires_at,
        permissions=api_key.permissions,
        rate_limit=api_key.rate_limit
    )

async def get_current_api_key(
    api_key: str = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db)
) -> APIKeyInfo:
    """Dependency to get and validate the current API key"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    
    key_info = await api_key_manager.validate_key(db, api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return key_info

@router.get("/keys/info", response_model=APIKeyInfo)
async def get_key_info(
    key_info: APIKeyInfo = Depends(get_current_api_key)
):
    """Get information about the current API key"""
    return key_info 