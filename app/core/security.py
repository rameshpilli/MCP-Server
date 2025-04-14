from typing import List, Optional, Tuple
from datetime import datetime, UTC
import secrets
from app.core.models import APIKey
from app.core.config import get_settings
from passlib.context import CryptContext
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Get settings instance
settings = get_settings()

class APIKeyManager:
    """Manages API key generation and validation."""
    
    async def create_key(
        self,
        db: AsyncSession,
        owner: str,
        expires_in_days: int = 365,
        permissions: List[str] = ["model:access"],
        rate_limit: str = "100/minute"
    ) -> APIKey:
        """Create a new API key and save it to the database.
        
        Args:
            db: Database session
            owner: Owner of the key
            expires_in_days: Number of days until key expires
            permissions: List of permissions
            rate_limit: Rate limit string (e.g. "100/minute")
            
        Returns:
            APIKey model instance with the plain key
        """
        # Generate key ID
        key_id = f"key_{secrets.token_urlsafe(8)}"
        
        # Calculate expiration date
        expires_at = datetime.now(UTC).replace(hour=23, minute=59, second=59)
        if expires_in_days:
            expires_at = expires_at.replace(day=expires_at.day + expires_in_days)
            
        # Generate the key
        plain_key, api_key = await self.generate_key(
            key_id=key_id,
            owner=owner,
            expires_at=expires_at,
            permissions=permissions,
            rate_limit=rate_limit
        )
        
        # Save to database
        db.add(api_key)
        await db.flush()
        
        # Create a copy of the API key with the plain key for response
        response_key = APIKey(
            key_id=api_key.key_id,
            key=plain_key,  # Use plain key for response
            owner=api_key.owner,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
            permissions=api_key.permissions,
            is_active=api_key.is_active,
            rate_limit=api_key.rate_limit
        )
        
        return response_key
    
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
        
        # Hash the key for storage
        hashed_key = create_api_key_hash(plain_key)
        
        # Create API key record
        api_key = APIKey(
            key_id=key_id,
            key=hashed_key,
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
        # Hash the provided key for comparison
        hashed_key = create_api_key_hash(key)
        
        # Use SQLAlchemy ORM for better type safety and maintainability
        result = await db.execute(
            select(APIKey).where(
                APIKey.key == hashed_key,
                APIKey.is_active == True
            )
        )
        api_key = result.scalar_one_or_none()
        
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