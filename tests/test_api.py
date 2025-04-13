import pytest
from fastapi.testclient import TestClient
from datetime import datetime, UTC, timedelta
import json
from sqlalchemy.ext.asyncio import AsyncSession
import time
import uuid
from sqlalchemy import select, delete
from app.main import app
from app.core.database import async_session_maker
from app.core.models import APIKey

@pytest.fixture
async def client():
    return TestClient(app)

@pytest.fixture
async def test_api_key():
    """Create a test API key for testing with a unique key each time"""
    unique_key = f"test_key_{uuid.uuid4().hex[:8]}"
    async with async_session_maker() as session:
        # Create new test key with unique identifier
        api_key = APIKey(
            key_id=f"test_key_id_{uuid.uuid4().hex[:8]}",
            key=unique_key,
            owner="test_user",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=1),
            permissions=["read", "write", "execute"],
            is_active=True,
            rate_limit="5/minute"
        )
        session.add(api_key)
        await session.commit()
        return api_key

@pytest.mark.asyncio
async def test_health_check(client: TestClient, db_session: AsyncSession):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_root_endpoint(client: TestClient, db_session: AsyncSession):
    """Test root endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_api_key_creation(client):
    """Test API key creation endpoint"""
    response = client.post(
        "/api/keys",
        json={
            "owner": "test_user",
            "expiry_days": 30,
            "permissions": ["read", "write"],
            "rate_limit": "5/minute"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "key" in data
    assert data["owner"] == "test_user"
    assert data["permissions"] == ["read", "write"]
    assert data["rate_limit"] == "5/minute"

@pytest.mark.asyncio
@pytest.mark.skip(reason="Tasks endpoint not implemented yet")
async def test_execute_task(client: TestClient, test_api_key: str, db_session: AsyncSession):
    """Test task execution."""
    response = client.post(
        "/api/tasks",
        json={
            "task_type": "test_task",
            "parameters": {"param1": "value1"}
        },
        headers={"X-API-Key": test_api_key}
    )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_model_registration_and_status(client, test_api_key):
    """Test model registration and status endpoints"""
    model_id = f"test_model_{uuid.uuid4().hex[:8]}"
    register_response = client.post(
        "/api/models/register",
        headers={"X-API-Key": test_api_key.key},
        json={
            "model_id": model_id,
            "name": "test_model",
            "version": "1.0",
            "description": "Test model for API",
            "backend": "local",
            "api_base": "http://localhost:8000",
            "config": {
                "type": "test",
                "version": "1.0"
            }
        }
    )
    assert register_response.status_code == 200
    data = register_response.json()
    assert data["status"] == "success"
    assert "model" in data
    assert data["model"]["model_id"] == model_id
    assert data["model"]["name"] == "test_model"
    assert data["model"]["backend"] == "local"

@pytest.mark.asyncio
async def test_invalid_api_key(client: TestClient, db_session: AsyncSession):
    """Test invalid API key handling."""
    response = client.get(
        "/api/keys/info",
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == 401
    error_data = response.json()
    assert "detail" in error_data
    assert "Invalid API key" in error_data["detail"]

@pytest.mark.asyncio
async def test_rate_limiting(client, test_api_key):
    """Test rate limiting functionality"""
    # First register a model to test status endpoint
    model_id = f"test_model_{uuid.uuid4().hex[:8]}"
    register_response = client.post(
        "/api/models/register",
        headers={"X-API-Key": test_api_key.key},
        json={
            "model_id": model_id,
            "name": "test_model",
            "version": "1.0",
            "description": "Test model for API",
            "backend": "local",
            "api_base": "http://localhost:8000"
        }
    )
    assert register_response.status_code == 200

    # Make requests up to the limit and check headers
    for i in range(5):
        response = client.get(
            f"/api/models/status/{model_id}",
            headers={"X-API-Key": test_api_key.key}
        )
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        assert int(response.headers["X-RateLimit-Limit"]) == 5
        assert int(response.headers["X-RateLimit-Remaining"]) == 4 - i
        
    # Next request should be rate limited
    response = client.get(
        f"/api/models/status/{model_id}",
        headers={"X-API-Key": test_api_key.key}
    )
    assert response.status_code == 429
    error_data = response.json()
    assert "detail" in error_data
    assert "rate limit exceeded" in error_data["detail"].lower()
    assert "reset_time" in error_data
    assert int(error_data["reset_time"]) <= 60  # Reset time should be within a minute
    assert "X-RateLimit-Reset" in response.headers
    assert int(response.headers["X-RateLimit-Reset"]) <= 60 