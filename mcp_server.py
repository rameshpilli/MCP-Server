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

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions
from app.api.mcp_server import mcp_server

# Import database and config
from app.core.database import init_db, get_engine
from app.core.config import get_settings
from app.core.logger import logger

settings = get_settings()


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[Dict[str, Any]]:
    """
    Set up MCP server lifecycle with database connection.
    
    This runs on server startup and shutdown to properly initialize
    and clean up database connections.
    """
    try:
        # Initialize database on startup
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
        
        # Yield any context needed in handlers
        yield {}
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {str(e)}")
        raise
    finally:
        # Clean up on shutdown
        try:
            logger.info("Closing database connections...")
            engine = get_engine()
            await engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")


async def run_mcp_server():
    """Run the MCP server using stdio protocol."""
    # Set server lifespan
    mcp_server.set_lifespan(server_lifespan)
    
    # Run as stdio server for Claude Desktop and MCP CLI
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="MCP Model Registry",
                server_version="1.0.0",
                capabilities=mcp_server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_server():
    """Entry point for running the server."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    run_server() 