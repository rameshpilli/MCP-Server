from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from app.core.dependencies import get_db
from app.core.config import Settings

router = APIRouter()
settings = Settings()

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.API_KEY_EXPIRY_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> Dict[str, Any]:
    """Verify the API key and return the decoded token."""
    try:
        payload = jwt.decode(api_key, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/auth/token")
async def create_api_key(
    owner: str,
    permissions: Optional[list[str]] = None,
    expires_in_days: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Create a new API key."""
    if permissions is None:
        permissions = ["read"]  # Default permission
    
    if expires_in_days is None:
        expires_in_days = settings.API_KEY_EXPIRY_DAYS
    
    token_data = {
        "sub": owner,
        "permissions": permissions,
        "created_at": datetime.utcnow().isoformat()
    }
    
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(days=expires_in_days)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "owner": owner,
        "permissions": permissions,
        "expires_in_days": expires_in_days
    }

@router.get("/auth/verify")
async def verify_token(
    token_data: Dict[str, Any] = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Verify an API key and return its details."""
    return {
        "valid": True,
        "owner": token_data["sub"],
        "permissions": token_data.get("permissions", []),
        "created_at": token_data.get("created_at"),
        "expires_at": datetime.fromtimestamp(token_data["exp"]).isoformat()
    }

@router.post("/auth/revoke")
async def revoke_token(
    token_data: Dict[str, Any] = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Revoke an API key."""
    # In a real implementation, we would add the token to a blacklist
    # or mark it as revoked in the database
    return {"status": "Token revoked successfully"} 