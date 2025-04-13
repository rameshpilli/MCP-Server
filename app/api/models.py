"""API endpoints for model management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
import logging
from pydantic import BaseModel

from app.core.database import get_db
from app.core.models import ModelRecord, APIKey
from app.core.auth import get_current_api_key

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

class ModelCreate(BaseModel):
    """Model for creating a new model record."""
    model_id: str
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    api_base: Optional[str] = None
    backend: str
    config: dict = {}

class ModelResponse(BaseModel):
    """Model for API responses."""
    model_id: str
    name: str
    description: Optional[str]
    version: Optional[str]
    api_base: Optional[str]
    backend: str
    created_at: datetime
    updated_at: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    last_used: Optional[datetime]
    average_latency: float
    config: dict

    class Config:
        from_attributes = True

@router.post("/models", response_model=ModelResponse)
async def create_model(
    model_data: ModelCreate,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key)
):
    """Create a new model."""
    # Check if model with same ID already exists
    existing = await db.execute(
        select(ModelRecord).where(ModelRecord.model_id == model_data.model_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail=f"Model with ID {model_data.model_id} already exists"
        )
    
    # Create new model record
    model = ModelRecord(
        model_id=model_data.model_id,
        name=model_data.name,
        description=model_data.description,
        version=model_data.version,
        api_base=model_data.api_base,
        backend=model_data.backend,
        config=model_data.config
    )
    
    db.add(model)
    await db.commit()
    await db.refresh(model)
    
    return model

@router.get("/models", response_model=List[ModelResponse])
async def list_models(
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key)
):
    """List all registered models."""
    result = await db.execute(select(ModelRecord))
    models = result.scalars().all()
    return list(models)

@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key)
):
    """Get details of a specific model."""
    result = await db.execute(
        select(ModelRecord).where(ModelRecord.model_id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )
    return model

@router.get("/models/status")
async def get_models_status(
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key)
):
    """Get general status of models."""
    result = await db.execute(select(ModelRecord))
    models = result.scalars().all()
    
    return {
        "total_models": len(models),
        "active_models": len([m for m in models if m.is_active]),
        "total_requests": sum(m.total_requests for m in models),
        "successful_requests": sum(m.successful_requests for m in models),
        "failed_requests": sum(m.failed_requests for m in models)
    }

@router.get(
    "/models/status/{model_id}",
    response_model=Dict[str, Any],
    summary="Get model status",
    description="Retrieve the status and metrics of a registered model",
    responses={
        200: {"description": "Model status retrieved successfully"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_model_status(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key)
) -> Dict[str, Any]:
    """
    Get the status and metrics of a registered model.
    
    Args:
        model_id: The unique identifier of the model
        db: Database session
        
    Returns:
        Dict containing model status and metrics
    
    Raises:
        HTTPException: If model not found (404) or other errors (500)
    """
    try:
        # Query the model
        query = select(ModelRecord).where(ModelRecord.model_id == model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model with ID '{model_id}' not found"
            )
        
        # Calculate time since last used
        last_used_str = "Never"
        if model.last_used:
            time_since = datetime.now(UTC) - model.last_used
            minutes = int(time_since.total_seconds() / 60)
            if minutes < 60:
                last_used_str = f"{minutes} minutes ago"
            else:
                hours = minutes // 60
                last_used_str = f"{hours} hours ago"
        
        # Calculate success rate
        success_rate = 0
        if model.total_requests > 0:
            success_rate = round((model.successful_requests / model.total_requests) * 100, 2)
        
        # Prepare response with all required fields
        return {
            "model_id": model.model_id,
            "name": model.name,
            "description": model.description,
            "version": model.version,
            "backend": model.backend,
            "status": "active" if model.is_active else "inactive",
            "metrics": {
                "total_requests": model.total_requests,
                "successful_requests": model.successful_requests,
                "failed_requests": model.failed_requests,
                "success_rate": success_rate,
                "last_used": last_used_str,
                "average_latency": round(model.average_latency or 0, 2),
                "total_tokens": model.total_tokens or 0
            },
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "config": model.config or {}
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        ) 