import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
import os
import asyncio
from datetime import datetime, timedelta, UTC
import logging
from typing import AsyncGenerator, Generator
import uuid

from app.main import app
from app.core.database import Base, get_db
from app.core.models import APIKey
from app.core.config import Settings, get_settings, StorageBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def set_test_env():
    """Set test environment variables before running tests"""
    os.environ["TESTING"] = "True"
    os.environ["STORAGE_BACKEND"] = "local"
    yield
    os.environ.pop("TESTING", None)
    os.environ.pop("STORAGE_BACKEND", None)

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Get test settings."""
    return Settings(
        TESTING=True,
        STORAGE_BACKEND=StorageBackend.LOCAL,
        TEST_DB_URL="sqlite+aiosqlite:///:memory:",
        DEBUG=True
    )

@pytest.fixture(scope="session")
async def test_engine(test_settings: Settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.TEST_DB_URL,
        echo=test_settings.DEBUG,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Get test database session."""
    session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )

    async with session_maker() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def test_api_key(db_session: AsyncSession) -> str:
    """Create a test API key."""
    key = APIKey(
        key="test_key",
        enabled=True,
        permissions=["*"],
        rate_limit=100,
        expires_at=datetime.now(UTC) + timedelta(days=30)
    )
    db_session.add(key)
    await db_session.commit()
    return key.key

@pytest.fixture
def client(test_settings: Settings, db_session: AsyncSession) -> Generator[TestClient, None, None]:
    """Create test client with overridden dependencies."""
    async def override_get_settings():
        return test_settings

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear() 