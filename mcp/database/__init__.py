"""
MCP Database Module

This module provides functionality for working with the MCP database,
including session management, database initialization, and model operations.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import logging
import os

from .config import DatabaseConfig

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
_engine = None
_session_factory = None

async def init_database(config: DatabaseConfig) -> None:
    """Initialize the database connection."""
    global _engine, _session_factory
    
    if _engine is None:
        logger.info(f"Initializing database with URL: {config.url}")
        connect_args = config.connect_args
        
        # Special handling for SQLite
        if config.url.startswith('sqlite'):
            # Ensure directory exists for SQLite file
            db_path = config.url.split('///')[1]
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # SQLite connect_args
            connect_args.update({"check_same_thread": False})
        
        _engine = create_async_engine(
            config.url,
            echo=config.echo,
            pool_pre_ping=True,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            connect_args=connect_args
        )
        
        _session_factory = async_sessionmaker(
            bind=_engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        logger.info("Database engine and session factory initialized")
        
        # Create tables
        from .models import Base
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database first.")
    
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e
    finally:
        await session.close()

async def close_database() -> None:
    """Close database connections."""
    global _engine
    
    if _engine is not None:
        logger.info("Closing database connections")
        await _engine.dispose()
        _engine = None
        logger.info("Database connections closed") 