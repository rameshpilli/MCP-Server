"""API package initialization."""

from fastapi import APIRouter
from . import query_endpoint, sources, auth, usage
from .keys import router as keys_router
from .models import router as models_router

# Create a combined router
router = APIRouter()

# Include all sub-routers
router.include_router(auth.router, tags=["Authentication"])
router.include_router(query_endpoint.router, tags=["Query"])
router.include_router(sources.router, tags=["Sources"])
router.include_router(usage.router, tags=["Usage"])
router.include_router(keys_router, tags=["API Keys"])
router.include_router(models_router, tags=["Models"])

__all__ = ["query_endpoint", "sources", "auth", "usage", "keys_router", "models_router", "router"] 