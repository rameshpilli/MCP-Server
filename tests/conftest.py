import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import AsyncGenerator, Generator
import uuid
import secrets

from app.main import app
from app.core.database import Base, get_db
from app.core.models import APIKey
from app.core.config import Settings, get_settings, StorageBackend
from app.core.startup import StartupValidator
from app.core.database import get_engine, get_session_factory
from app.core.security import APIKeyManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set test environment variables before running tests"""
    os.environ["TESTING"] = "True"
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["TEST_DB_URL"] = "sqlite+aiosqlite:///:memory:"
    yield
    os.environ.pop("TESTING", None)
    os.environ.pop("STORAGE_BACKEND", None)
    os.environ.pop("TEST_DB_URL", None)

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def settings() -> Settings:
    """Create test settings"""
    settings = get_settings()
    settings.TESTING = True
    settings.TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
    return settings

@pytest.fixture
async def test_engine(settings):
    """Create a test database engine."""
    engine = create_async_engine(
        settings.TEST_DB_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        isolation_level="AUTOCOMMIT"
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)  # Drop all tables first
        await conn.run_sync(Base.metadata.create_all)  # Create all tables
    
    try:
        yield engine
    finally:
        await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    session = AsyncSession(bind=test_engine, expire_on_commit=False)
    try:
        yield session
    finally:
        await session.close()

@pytest.fixture
async def test_api_key(db_session: AsyncSession) -> tuple[APIKey, str]:
    """Create a test API key.
    
    Returns:
        Tuple of (APIKey record, plain key string)
    """
    key_id = f"test_key_{secrets.token_hex(8)}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    
    # Create API key manager and generate key
    api_key_manager = APIKeyManager()
    plain_key, api_key = await api_key_manager.generate_key(
        key_id=key_id,
        owner="test@example.com",
        expires_at=expires_at,
        permissions=["*"],  # All permissions for testing
        rate_limit="100/minute"  # Rate limit for testing
    )
    
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)
    return api_key, plain_key

@pytest_asyncio.fixture
async def client(settings: Settings, db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client"""
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        try:
            yield db_session
        finally:
            pass  # Session cleanup is handled by db_session fixture
    
    def override_get_settings() -> Settings:
        return settings

    # Override get_engine to return test engine
    def override_get_engine():
        return db_session.bind
    
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_engine] = override_get_engine
    
    # Disable startup validation for tests
    app.router.on_startup.clear()
    
    async with AsyncClient(app=app, base_url="http://testserver") as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture
async def startup_validator(settings: Settings) -> StartupValidator:
    """Create startup validator for testing"""
    return StartupValidator(config=settings) 