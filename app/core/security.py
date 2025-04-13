from typing import List, Optional, Tuple
from datetime import datetime, UTC
import secrets
from app.core.models import APIKey
from app.core.config import get_settings
from passlib.context import CryptContext
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession

# Get settings instance
settings = get_settings()

class APIKeyManager:
    """Manages API key generation and validation."""
    
    async def generate_key(
        self,
        key_id: str,
        owner: str,
        expires_at: datetime,
        permissions: List[str],
        rate_limit: Optional[str] = None
    ) -> Tuple[str, APIKey]:
        """Generate a new API key.
        
        Args:
            key_id: Unique identifier for the key
            owner: Owner of the key
            expires_at: Expiration datetime
            permissions: List of permissions
            rate_limit: Optional rate limit string (e.g. "100/minute")
            
        Returns:
            Tuple of (plain_key, APIKey model instance)
        """
        # Generate a random key
        plain_key = secrets.token_urlsafe(32)
        
        # Create API key record
        api_key = APIKey(
            key_id=key_id,
            key=plain_key,
            owner=owner,
            created_at=datetime.now(UTC),
            expires_at=expires_at,
            permissions=permissions,
            is_active=True,
            rate_limit=rate_limit or "100/minute"
        )
        
        return plain_key, api_key
    
    async def validate_key(
        self,
        db: AsyncSession,
        key: str
    ) -> Optional[APIKey]:
        """Validate an API key.
        
        Args:
            db: Database session
            key: API key to validate
            
        Returns:
            APIKey if valid, None if invalid
        """
        result = await db.execute(
            "SELECT * FROM api_keys WHERE key = :key AND is_active = TRUE",
            {"key": key}
        )
        api_key = result.fetchone()
        
        if not api_key:
            return None
            
        if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
            return None
            
        return api_key

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_api_key_hash(api_key: str) -> str:
    """Create a hash of the API key using SHA-256"""
    return hashlib.sha256(api_key.encode()).hexdigest() 