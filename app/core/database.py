import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
import asyncpg
from app.core.config import get_settings
from app.core.logger import logger

# Base class for all models
Base = declarative_base()

# Get settings instance
settings = get_settings()

async def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    try:
        # Connect to default PostgreSQL database to check if our DB exists
        conn = await asyncpg.connect(
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database='postgres'  # Connect to default postgres database
        )

        # Check if our database exists
        result = await conn.fetchrow(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            settings.DB_NAME
        )

        if not result:
            # Database doesn't exist, create it
            await conn.execute(f'CREATE DATABASE "{settings.DB_NAME}"')
            logger.info(f"Created database {settings.DB_NAME}")
        
        await conn.close()
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise

# Create engine based on settings
def get_engine():
    """Get database engine based on settings"""
    if settings.TESTING:
        return create_async_engine(
            settings.TEST_DB_URL,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=settings.DEBUG
        )
    else:
        return create_async_engine(
            settings.get_db_url(),
            echo=settings.DEBUG
        )

# Create session factory
def get_session_factory():
    """Get session factory based on settings"""
    return async_sessionmaker(
        get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False
    )

async def init_db():
    """Initialize database and create all tables"""
    if not settings.TESTING:
        # Create database if it doesn't exist
        await create_database_if_not_exists()
    
    # Create engine and tables
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Verify connection
        try:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close() 