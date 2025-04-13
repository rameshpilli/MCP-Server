from functools import lru_cache
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.database import get_db

@lru_cache()
def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

# Export get_db from database module
__all__ = ['get_settings', 'get_db'] 