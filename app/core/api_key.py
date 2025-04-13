from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import APIKey
from app.core.security import create_api_key_hash
import secrets
import string

class APIKeyManager:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_api_key(
        self, 
        owner: str, 
        expiry_days: Optional[int] = None,
        permissions: List[str] = [],
        rate_limit: str = "5/minute"
    ) -> tuple[str, APIKey]:
        """Create a new API key"""
        # Generate a random API key
        alphabet = string.ascii_letters + string.digits
        api_key = ''.join(secrets.choice(alphabet) for _ in range(32))
        
        # Calculate expiry date if provided
        expires_at = None
        if expiry_days is not None:
            expires_at = datetime.utcnow() + timedelta(days=expiry_days)

        # Create API key record
        api_key_record = APIKey(
            key=create_api_key_hash(api_key),
            owner=owner,
            expires_at=expires_at,
            permissions=permissions,
            rate_limit=rate_limit
        )
        
        self.db.add(api_key_record)
        await self.db.commit()
        await self.db.refresh(api_key_record)
        
        return api_key, api_key_record

    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key and return the associated record if valid"""
        if not api_key:
            return None
            
        # Get API key record
        hashed_key = create_api_key_hash(api_key)
        query = select(APIKey).where(APIKey.key == hashed_key)
        result = await self.db.execute(query)
        api_key_record = result.scalar_one_or_none()
        
        if not api_key_record:
            return None
            
        # Check if expired
        if api_key_record.expires_at and api_key_record.expires_at < datetime.utcnow():
            return None
            
        return api_key_record 