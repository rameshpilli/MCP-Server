#!/usr/bin/env python3

import sys
import os
from datetime import datetime, timedelta, UTC
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import asyncio

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.security import APIKeyManager
from app.core.models import Base
from app.core.config import get_settings

async def create_admin_key():
    """Create an admin API key for initial setup."""
    settings = get_settings()
    
    # Create async engine
    engine = create_async_engine(settings.ASYNC_DATABASE_URL)
    
    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # Create API key manager
        api_key_manager = APIKeyManager()
        
        # Generate admin key that never expires
        plain_key, api_key = await api_key_manager.generate_key(
            key_id="admin_key",
            owner="admin",
            expires_at=None,  # Never expires
            permissions=["admin", "read", "write"],  # All permissions
            rate_limit="1000/minute"  # High rate limit for admin
        )
        
        # Save to database
        session.add(api_key)
        await session.commit()
        
        print("\n✨ Admin API key created successfully!")
        print("\n⚠️  IMPORTANT: Save this key securely. It will not be shown again!")
        print("\nTo use this key, set it in your environment:")
        print(f"\nexport MCP_API_KEY='{plain_key}'")
        print("\nOr add it to your .env file:")
        print(f"\nMCP_API_KEY='{plain_key}'")

if __name__ == "__main__":
    asyncio.run(create_admin_key()) 