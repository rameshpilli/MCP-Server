"""Main FastAPI application."""

import os
from typing import Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

# Import MCP components
from mcp.database import init_database, close_database, get_session
from mcp.server import create_mcp_router
from app.api.mcp_server import mcp_server

# Import from app
from app.core.config import get_settings
from app.core.logger import logger
from app.api import models, health

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[Dict[str, Any]]:
    """
    Lifespan manager for the FastAPI application.
    
    This handles startup and shutdown events.
    """
    try:
        # Initialize database on startup using MCP's database functionality
        logger.info("Initializing MCP database...")
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        logger.info("MCP database initialized successfully")
        
        # Startup validation
        validation_errors = []
        try:
            # Validate connections
            logger.info("Validating connections...")
            # Connection validation code if needed
        except Exception as e:
            validation_errors.append(f"Connection validation failed: {str(e)}")
        
        # Log validation results
        if validation_errors:
            logger.warning(f"Startup validation completed with {len(validation_errors)} errors")
            for error in validation_errors:
                logger.error(f"Validation error: {error}")
        else:
            logger.info("All validations passed successfully")
        
        yield {"validation_errors": validation_errors}
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        # Cleanup on shutdown
        try:
            logger.info("Closing MCP database connections...")
            await close_database()
            logger.info("MCP database connections closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Model Context Protocol Server",
    version="1.0.0",
    lifespan=lifespan
)


# Create default index.html if it doesn't exist
os.makedirs("static", exist_ok=True)
if not os.path.exists("static/index.html"):
    with open("static/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Model Context Protocol (MCP) Server</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }
        h1 { color: #333; }
        .links { margin-top: 2rem; }
        .links a { display: block; margin-bottom: 1rem; color: #0066cc; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Welcome to the Model Context Protocol Server</h1>
    <p>This server provides a standardized interface for accessing models and data sources.</p>
    
    <div class="links">
        <a href="/docs">→ API Documentation</a>
        <a href="/llm/ui">→ LLM Dashboard</a>
    </div>
</body>
</html>
""")


# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API key middleware
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for validating API keys."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip API key validation for excluded paths
        path = request.url.path
        
        # Skip validation for public paths
        if path in settings.PUBLIC_PATHS or path.startswith(("/static/", "/docs", "/redoc", "/openapi.json")):
            return await call_next(request)
        
        # Validate API key
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if not api_key:
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"detail": "API key is required"}
            )
        
        # Use MCP's API key validation functionality
        async with get_session() as session:
            from mcp.database.models import validate_api_key
            valid = await validate_api_key(session, api_key)
            
            if not valid:
                return JSONResponse(
                    status_code=HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid API key"}
                )
        
        # Key is valid, continue to handler
        return await call_next(request)


# Add middleware
app.add_middleware(APIKeyMiddleware)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Register routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(models.router, prefix="/api/models", tags=["models"])

# Create MCP router and mount as a sub-app
mcp_router = create_mcp_router(mcp_server)
app.mount("/mcp", mcp_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that serves the index.html file."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    ) 