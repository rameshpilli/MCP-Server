#!/usr/bin/env python
"""
MCP Server entrypoint for Model Context Protocol.

This server exposes models and data sources through the Model Context Protocol (MCP).
It can be used directly with Claude and other MCP-compatible clients.

Run with:
    python mcp_server.py
    
Or with MCP CLI:
    mcp run mcp_server.py
    
To install in Claude Desktop:
    mcp install mcp_server.py
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Dict, Any, AsyncIterator
from pathlib import Path

# Load environment variables before imports
load_dotenv()

# Make sure the app package is in the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import from our mock MCP implementation
from fastapi import FastAPI
from mcp.database import init_database, close_database
from app.api.mcp_server import mcp_server

# Import config
from app.core.config import get_settings
from app.core.logger import logger

settings = get_settings()


@asynccontextmanager
async def server_lifespan(server) -> AsyncIterator[Dict[str, Any]]:
    """
    Set up MCP server lifecycle with database connection.
    
    This runs on server startup and shutdown to properly initialize
    and clean up database connections.
    """
    try:
        # Initialize database on startup using MCP's database functionality
        logger.info("Initializing MCP database...")
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        logger.info("MCP database initialized successfully")
        
        # Yield any context needed in handlers
        yield {}
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {str(e)}")
        raise
    finally:
        # Clean up on shutdown using MCP's database functionality
        try:
            logger.info("Closing MCP database connections...")
            await close_database()
            logger.info("MCP database connections closed")
        except Exception as e:
            logger.error(f"Error closing MCP database connections: {str(e)}")


async def run_mcp_server():
    """Run the MCP server as a FastAPI app."""
    # Create FastAPI app
    app = FastAPI(
        title="MCP Model Registry",
        description="Model Context Protocol Server",
        version="1.0.0",
        lifespan=server_lifespan
    )
    
    # Get MCP router from server
    router = mcp_server.get_router()
    
    # Include MCP router
    app.include_router(router, prefix="/mcp")
    
    # Return the FastAPI app
    return app


def run_server():
    """Entry point for running the server."""
    # For standalone execution, run with uvicorn
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "mcp_server:run_mcp_server",
        host=host,
        port=port,
        factory=True,
        reload=settings.DEBUG
    )


if __name__ == "__main__":
    run_server() 