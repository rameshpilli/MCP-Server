"""API endpoints for model management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC, timedelta
import logging
from pydantic import BaseModel
import json

from app.core.database import get_db
from app.core.models import ModelRecord, APIKey
from app.core.auth import get_current_api_key, APIKeyManager
from app.core.config import ModelBackend, get_settings

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
    description: Optional[str] = None
    version: Optional[str] = None
    api_base: Optional[str] = None
    backend: str
    created_at: datetime
    updated_at: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    last_used: Optional[datetime] = None
    average_latency: float = 0.0
    config: dict = {}
    api_key: Optional[str] = None  # Make api_key optional

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

@router.post("/models/register", response_model=ModelResponse)
async def register_model(
    model_data: ModelCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new model and generate an API key."""
    try:
        # Validate backend type
        try:
            backend = ModelBackend(model_data.backend.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid backend type. Must be one of: {', '.join([b.value for b in ModelBackend])}"
            )

        # Clean up model ID by replacing spaces with hyphens and making it lowercase
        model_id = model_data.model_id.strip().replace(" ", "-").lower()

        # Check if model with same ID already exists
        existing = await db.execute(
            select(ModelRecord).where(ModelRecord.model_id == model_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"Model with ID {model_id} already exists"
            )
        
        # Create new model record
        model = ModelRecord(
            model_id=model_id,  # Use the cleaned model ID
            name=model_data.name,
            description=model_data.description,
            version=model_data.version,
            api_base=model_data.api_base,
            backend=backend,
            config=model_data.config,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_tokens=0,
            average_latency=0.0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        
        db.add(model)
        await db.flush()  # Flush to get the model ID

        # Generate API key for the model
        api_key_manager = APIKeyManager()
        api_key = await api_key_manager.create_key(
            db=db,
            owner=model_data.name,
            expires_in_days=365,  # 1 year expiry
            permissions=["model:access"],
            rate_limit="100/minute"
        )

        if not api_key or not api_key.key:
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Failed to generate API key for model"
            )

        # Log the API key for debugging
        logger.info(f"Generated API key for model {model_id}: {api_key.key}")

        await db.commit()
        await db.refresh(model)
        
        # Create response with both model data and API key
        response_data = {
            "model_id": model.model_id,
            "name": model.name,
            "description": model.description,
            "version": model.version,
            "api_base": model.api_base,
            "backend": model.backend,
            "config": model.config,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "total_requests": model.total_requests,
            "successful_requests": model.successful_requests,
            "failed_requests": model.failed_requests,
            "total_tokens": model.total_tokens,
            "last_used": model.last_used,
            "average_latency": model.average_latency,
            "api_key": api_key.key  # This will be the plain API key
        }

        # Log the response data for debugging (without the API key for security)
        log_data = {**response_data}
        log_data["api_key"] = "*** REDACTED ***"
        logger.info(f"Sending model registration response: {json.dumps(log_data)}")
        
        # Create and return the response model
        response = ModelResponse(**response_data)
        logger.info(f"Final response object has api_key: {response.api_key is not None}")
        
        return response

    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register model: {str(e)}"
        )

@router.get("/models", response_model=List[ModelResponse])
async def list_models(
    db: AsyncSession = Depends(get_db)
):
    """List all registered models."""
    try:
        result = await db.execute(select(ModelRecord))
        models = result.scalars().all()
        
        # Convert models to response format without API keys
        model_responses = []
        for model in models:
            model_dict = {
                "model_id": model.model_id,
                "name": model.name,
                "description": model.description,
                "version": model.version,
                "api_base": model.api_base,
                "backend": model.backend,
                "config": model.config,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "total_requests": model.total_requests,
                "successful_requests": model.successful_requests,
                "failed_requests": model.failed_requests,
                "total_tokens": model.total_tokens,
                "last_used": model.last_used,
                "average_latency": model.average_latency
            }
            model_responses.append(ModelResponse(**model_dict))
        
        return model_responses
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )

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

@router.post("/models/{model_id}/regenerate-key", response_model=Dict[str, str])
async def regenerate_api_key(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Regenerate API key for an existing model."""
    try:
        # Check if model exists
        result = await db.execute(
            select(ModelRecord).where(ModelRecord.model_id == model_id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        # Generate new API key
        api_key_manager = APIKeyManager()
        api_key = await api_key_manager.create_key(
            db,
            owner=model.name,
            expires_in_days=365,  # 1 year expiry
            permissions=["model:access"],
            rate_limit="100/minute"
        )

        if not api_key or not api_key.key:
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Failed to generate API key for model"
            )

        # Log the new API key for debugging
        logger.info(f"Regenerated API key for model {model_id}: {api_key.key}")

        await db.commit()
        
        return {"api_key": api_key.key}

    except Exception as e:
        logger.error(f"Error regenerating API key: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate API key: {str(e)}"
        ) 