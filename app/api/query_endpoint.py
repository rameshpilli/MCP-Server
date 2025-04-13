from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, Security, APIRouter, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, UTC
import os
import time
import secrets
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.core.metadata_manager import MetadataManager
from app.core.router import IntelligentRouter
from app.core.database import get_db
from app.core.models import ModelRecord
from app.core.model_client import ModelClient, ModelContext
from app.core.logger import logger
from app.core.auth import get_current_model, check_model_permissions
from app.core.config import ModelBackend, get_settings

# Get settings instance
settings = get_settings()

# Create router instead of app
router = APIRouter()

# Templates can be initialized in main.py
templates = Jinja2Templates(directory="templates")

class ModelRegistration(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str
    parameters: Optional[dict] = None

class QueryResponse(BaseModel):
    query_id: str
    result: dict
    execution_time: float
    source: str
    row_count: int

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/models/register")
async def register_model(
    model_id: str = Body(...),
    name: str = Body(...),
    version: str = Body(...),
    description: str = Body(None),
    backend: str = Body(ModelBackend.LOCAL.value),  # Accept string value
    api_base: str = Body(None),
    db: AsyncSession = Depends(get_db)
):
    """Register a new model."""
    try:
        # Convert backend string to enum
        try:
            backend_enum = ModelBackend(backend.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid backend type. Must be one of: {[b.value for b in ModelBackend]}"
            )

        # Create metadata
        metadata = {
            "description": description,
            "config": {
                "name": name,
                "version": version
            }
        }
        
        model = ModelRecord(
            model_id=model_id,
            name=name,
            version=version,
            backend=backend_enum,
            api_base=api_base,
            metadata=metadata,
            is_active=True  # Set is_active to True by default
        )
        
        db.add(model)
        await db.commit()
        await db.refresh(model)
        
        return {
            "status": "success",
            "message": f"Model {model_id} registered successfully",
            "model": {
                "model_id": model.model_id,
                "name": name,
                "version": version,
                "description": description,
                "backend": backend_enum.value,
                "api_base": api_base,
                "is_active": model.is_active
            }
        }
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"Model with ID {model_id} already exists"
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to register model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register model: {str(e)}"
        )

async def update_model_stats(
    model_id: str,
    source_id: str,
    execution_time: float,
    success: bool,
    query_type: Optional[str] = None
):
    """Background task to update model statistics"""
    try:
        async with ModelClient(model_id=model_id) as client:
            await client._update_stats(success, execution_time, {'source': source_id})
    except Exception as e:
        logger.error(f"Failed to update model stats: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    model: dict = Depends(get_current_model)
):
    """Execute a query and return results"""
    try:
        # Record start time
        start_time = datetime.now()
        
        # TODO: Implement actual query execution
        result = {
            "data": [{"id": 1, "name": "test"}],
            "metadata": {"source": "test_db"}
        }
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query_id="test_query_id",
            result=result,
            execution_time=execution_time,
            source="test_db",
            row_count=1
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )

@router.get("/sources")
async def list_sources(
    model: ModelRecord = Depends(check_model_permissions(["sources:list"])),
    db: AsyncSession = Depends(get_db)
):
    """List available data sources and their tables"""
    metadata_manager = MetadataManager(settings.SQLITE_DB_URL)
    return await metadata_manager.get_all_sources()

@router.get("/query_history")
async def get_query_history(
    limit: int = 100,
    model: ModelRecord = Depends(check_model_permissions(["history:view"])),
    db: AsyncSession = Depends(get_db)
):
    """Get query history for a model"""
    try:
        async with ModelClient(model_id=model.id) as client:
            context = await client.get_context()
            return {
                'recent_queries': context.recent_queries[-limit:],
                'stats': await client.get_stats()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sources/health")
async def get_source_health(
    model: ModelRecord = Depends(check_model_permissions(["sources:health"])),
    db: AsyncSession = Depends(get_db)
):
    """Get health status of all data sources"""
    router = IntelligentRouter(db)
    return {
        source_id: {
            "is_available": health.is_available,
            "error_count": health.error_count,
            "last_error": health.last_error,
            "last_checked": health.last_checked,
            "recovery_attempts": health.recovery_attempts,
            "last_recovery": health.last_recovery
        }
        for source_id, health in router.source_health.items()
    }

@router.post("/sources/{source_id}/reset")
async def reset_source(
    source_id: str,
    model: ModelRecord = Depends(check_model_permissions(["sources:manage"])),
    db: AsyncSession = Depends(get_db)
):
    """Reset health status for a data source"""
    router = IntelligentRouter(db)
    await router.reset_source_health(source_id)
    return {"status": "success", "message": f"Health status reset for source {source_id}"}

@router.get("/models/{model_id}/context")
async def get_model_context(
    model_id: str,
    current_model: ModelRecord = Depends(get_current_model)
):
    """Get current context for a model"""
    if current_model.id != model_id and "admin" not in current_model.config.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this model's context"
        )
        
    try:
        async with ModelClient(model_id=model_id) as client:
            return await client.get_context()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/context/clear")
async def clear_model_context(
    model_id: str,
    current_model: ModelRecord = Depends(get_current_model)
):
    """Clear context for a model"""
    if current_model.id != model_id and "admin" not in current_model.config.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail="Not authorized to clear this model's context"
        )
        
    try:
        async with ModelClient(model_id=model_id) as client:
            await client.clear_context()
            return {"status": "success", "message": "Context cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 