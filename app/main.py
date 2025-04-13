import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
import logging

from app.core.startup import StartupValidator
from app.core.logger import logger
from app.core.middleware import UsageTrackingMiddleware, APIKeyMiddleware, get_api_key_from_header
from app.api import router as api_router, health, keys, models
from app.core.config import Settings, get_settings
from app.core.database import init_db, get_db, get_engine, get_session_factory, Base
from app.core.auth import api_key_manager

settings = get_settings()
startup_validator = StartupValidator(config=settings)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Ensure static and templates directories exist
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Create a default index.html if it doesn't exist
index_html = TEMPLATES_DIR / "index.html"
if not index_html.exists():
    index_html.write_text("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Context Protocol</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .content { max-width: 800px; margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="content">
            <h1>Welcome to Model Context Protocol</h1>
            <p>A platform for managing and controlling ML model deployments.</p>
            <p>Visit <a href="/api/docs">API documentation</a> to learn more.</p>
        </div>
    </body>
    </html>
    """)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Initialize database
    await init_db()
    
    # Run startup checks
    checks = await startup_validator.run_all_checks()
    if not all(checks.values()):
        logger.error("Startup checks failed", checks=checks)
        # You might want to exit here depending on your requirements
    
    yield
    
    # Cleanup
    logger.info("Shutting down application")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs" if not settings.PRODUCTION else None,
    redoc_url="/redoc" if not settings.PRODUCTION else None,
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add usage tracking middleware
app.add_middleware(UsageTrackingMiddleware)

# Add API key middleware
app.add_middleware(APIKeyMiddleware)

# Include API router with prefix
app.include_router(api_router, prefix="/api", dependencies=[Depends(get_db)])
app.include_router(health.router, tags=["health"])
app.include_router(keys.router, prefix="/api/keys", tags=["api-keys"])
app.include_router(models.router, prefix="/api/models", tags=["models"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the index page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        checks = await startup_validator.run_all_checks()
        all_passed = all(checks.values())
        return {
            "status": "healthy" if all_passed else "unhealthy",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception", extra={
        "error": str(exc),
        "method": request.method,
        "path": request.url.path
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    try:
        # Initialize database
        await init_db()
        
        # Run startup validation
        validator = StartupValidator(settings)
        results = await validator.run_all_checks()
        
        # Check if any validation failed
        failed_checks = [check[0] for check in results if not check[1]]
        if failed_checks:
            raise HTTPException(
                status_code=500,
                detail=f"System validation failed. Failed checks: {failed_checks}"
            )
            
        logger.info("✓ Application startup complete")
    except Exception as e:
        logger.error(f"✗ Application startup failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Application startup failed: {str(e)}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    try:
        # Close database connections
        engine = get_engine()
        await engine.dispose()
        logger.info("✓ Application shutdown complete")
    except Exception as e:
        logger.error(f"✗ Application shutdown failed: {str(e)}") 