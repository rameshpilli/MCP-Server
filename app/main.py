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

from app.core.startup import StartupValidator
from app.core.logger import logger
from app.core.middleware import UsageTrackingMiddleware, APIKeyMiddleware, get_api_key_from_header
from app.api import router as api_router
from app.core.config import Settings, get_settings
from app.core.database import init_db, get_db, async_session_maker
from app.core.auth import api_key_manager

startup_validator = StartupValidator()
settings = get_settings()

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
        <title>Model Control Platform</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .content { max-width: 800px; margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="content">
            <h1>Welcome to Model Control Platform</h1>
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
    title="Model Control Platform",
    description="A platform for managing and controlling ML model deployments",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

class DatabaseSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        async with async_session_maker() as session:
            request.state.db = session
            response = await call_next(request)
            return response

# Add middleware in correct order
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware in correct order (from innermost to outermost)
app.add_middleware(UsageTrackingMiddleware)  # Outermost - tracks all requests
app.add_middleware(APIKeyMiddleware)  # Then validate API keys
app.add_middleware(DatabaseSessionMiddleware)  # Innermost - provides DB session

# Include API router with prefix
app.include_router(api_router, prefix="/api", dependencies=[Depends(get_db)])

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
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": request.url.path,
            "method": request.method
        }
    ) 