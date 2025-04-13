import pytest
from datetime import datetime, UTC, timedelta
import json
import uuid
import asyncio
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient
from app.core.models import APIKey
from app.main import app
from app.core.database import async_sessionmaker
import time

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "mode": "test"}

@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Model Control Platform</title>" in response.text

@pytest.mark.asyncio
async def test_api_key_creation(client: AsyncClient, db_session: AsyncSession):
    # Create a new API key
    key_id = f"test_key_id_{uuid.uuid4().hex[:8]}"
    response = await client.post(
        "/api/keys",
        json={
            "key_id": key_id,
            "owner": "test@example.com",
            "permissions": ["read", "write"],
            "rate_limit": "5/minute",
            "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat()
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert "key" in data
    assert data["key_id"] == key_id
    assert data["owner"] == "test@example.com"
    assert data["permissions"] == ["read", "write"]
    assert data["rate_limit"] == "5/minute"

    # Verify the key exists in the database
    result = await db_session.execute(select(APIKey).where(APIKey.key_id == key_id))
    api_key = result.scalar_one()
    assert api_key is not None
    assert api_key.owner == "test@example.com"

@pytest.mark.asyncio
@pytest.mark.skip(reason="Tasks endpoint not implemented yet")
async def test_execute_task(client: AsyncClient, test_api_key: tuple[APIKey, str], db_session: AsyncSession):
    """Test task execution."""
    api_key, plain_key = test_api_key
    response = await client.post(
        "/api/tasks",
        json={
            "task_type": "test_task",
            "parameters": {"param1": "value1"}
        },
        headers={"X-API-Key": plain_key}
    )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_model_registration_and_status(
    client: AsyncClient,
    test_api_key: tuple[APIKey, str],
    db_session: AsyncSession
):
    """Test model registration and status check."""
    api_key, plain_key = test_api_key
    
    # Register a new model
    response = await client.post(
        "/api/models",
        headers={"X-API-Key": plain_key},
        json={
            "name": "test-model",
            "version": "1.0",
            "description": "Test model"
        }
    )
    assert response.status_code == 201
    
    # Check model status
    response = await client.get(
        f"/api/models/test-model",
        headers={"X-API-Key": plain_key}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "registered"

@pytest.mark.asyncio
async def test_invalid_api_key(client: AsyncClient):
    """Test invalid API key returns 401."""
    response = await client.get(
        "/api/models/test-model",
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]

@pytest.mark.asyncio
async def test_rate_limiting(
    client: AsyncClient,
    test_api_key: tuple[APIKey, str],
    db_session: AsyncSession
):
    """Test rate limiting."""
    api_key, plain_key = test_api_key
    
    # Make multiple requests
    for _ in range(3):
        response = await client.get(
            "/api/models/test-model",
            headers={"X-API-Key": plain_key}
        )
        assert response.status_code == 200
    
    # Fourth request should be rate limited
    response = await client.get(
        "/api/models/test-model",
        headers={"X-API-Key": plain_key}
    )
    assert response.status_code == 429
    assert "Too many requests" in response.json()["detail"] 