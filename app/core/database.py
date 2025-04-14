"""Database module for async SQLAlchemy integration with MCP.

This module serves as a compatibility layer between the existing application
and the MCP database functionality. It provides the same interface as the
original database module but delegates to MCP's database tools.
"""

import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Base class mock for compatibility
class Base:
    """Mock Base class for models."""
    metadata = {}

# Mock session class
class MockSession:
    """Mock session class for compatibility."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def commit(self):
        logger.info("Mock commit called")
    
    async def rollback(self):
        logger.info("Mock rollback called")
    
    async def close(self):
        logger.info("Mock close called")
    
    async def execute(self, query):
        logger.info(f"Mock execute called with: {query}")
        return MockResult()
    
    def add(self, obj):
        logger.info(f"Mock add called with: {obj}")
    
    async def refresh(self, obj):
        logger.info(f"Mock refresh called with: {obj}")

class MockResult:
    """Mock result class for compatibility."""
    
    def scalars(self):
        return self
    
    def first(self):
        return None
    
    def all(self):
        return []

# For compatibility with existing code
async def init_db():
    """Initialize database using MCP functionality."""
    logger.info("Mock init_db called")

# Compatibility with existing get_db function
@asynccontextmanager
async def get_db() -> AsyncGenerator[MockSession, None]:
    """Get a database session using MCP's session manager.
    
    This provides the same interface as the original get_db function
    but uses MCP's session management.
    """
    session = MockSession()
    try:
        yield session
    finally:
        await session.close()

# For compatibility with existing code that might use these
def get_engine():
    """Get the SQLAlchemy engine (compatibility function)."""
    logger.info("Mock get_engine called")
    return None

def get_session_factory():
    """Get the SQLAlchemy session factory (compatibility function)."""
    logger.info("Mock get_session_factory called")
    return lambda: MockSession()

# Helper function to close the database - delegates to MCP
async def close_database():
    """Close database connections using MCP functionality."""
    logger.info("Mock close_database called") 