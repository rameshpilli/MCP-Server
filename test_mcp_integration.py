#!/usr/bin/env python3
"""
Test MCP Integration

This script tests the MCP database and server integration to ensure everything
is working correctly with our mock implementations.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import the MCP package
# from dotenv import load_dotenv
from mcp.database import init_database, get_session, close_database
from mcp.database.models import Model, DataSource, get_model, create_model, get_data_source_by_name
from app.core.config import get_settings
from app.core.logger import logger

# Load environment variables manually if needed
def load_env_from_file(env_file=".env"):
    """Load environment variables from a file."""
    if not os.path.exists(env_file):
        logger.warning(f"Environment file {env_file} not found, using existing environment")
        return
    
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value

# Load environment variables
try:
    load_env_from_file()
except Exception as e:
    logger.warning(f"Error loading environment variables: {str(e)}")

# Get settings
settings = get_settings()

async def test_database_connection():
    """Test the MCP database connection."""
    logger.info("Testing MCP database connection...")
    
    try:
        # Initialize the database
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        logger.info("✓ Database connection successful")
        
        # Check session functionality
        async with get_session() as session:
            logger.info("✓ Session creation successful")
            
        logger.info("Database connection test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {str(e)}")
        return False
    finally:
        await close_database()

async def test_model_operations():
    """Test model CRUD operations."""
    logger.info("Testing model operations...")
    
    try:
        # Initialize the database
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        
        # Create a test model
        test_model = Model(
            id="test-model-1",
            name="Test Model",
            description="A test model for MCP integration",
            backend="openai",
            version="1.0.0",
            api_base="https://api.openai.com/v1",
            configuration={"model_name": "gpt-4"},
            is_active=True,
            metrics={
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "average_latency": 0.0
            }
        )
        
        # Save the model
        async with get_session() as session:
            # Check if model already exists
            existing = await get_model(session, "test-model-1")
            if existing:
                logger.info("Test model already exists, skipping creation")
            else:
                await create_model(session, test_model)
                logger.info("✓ Model creation successful")
            
            # Retrieve the model
            model = await get_model(session, "test-model-1")
            if model:
                logger.info(f"✓ Model retrieval successful: {model.name}")
            else:
                logger.error("✗ Model retrieval failed")
                return False
            
        logger.info("Model operations test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Model operations test failed: {str(e)}")
        return False
    finally:
        await close_database()

async def test_data_source_operations():
    """Test data source operations."""
    logger.info("Testing data source operations...")
    
    try:
        # Initialize the database
        db_config = settings.get_mcp_database_config()
        await init_database(db_config)
        
        # Create a test data source
        snowflake_config = {
            "account": "your-account",
            "user": "your-user",
            "password": "your-password",
            "warehouse": "your-warehouse",
            "database": "your-database",
            "schema": "your-schema"
        }
        
        test_source = DataSource(
            id="test-snowflake-1",
            name="Test Snowflake",
            type="snowflake",
            description="A test Snowflake data source",
            configuration=snowflake_config,
            is_active=True
        )
        
        # Save the data source
        async with get_session() as session:
            # Check if source already exists
            existing = await get_data_source_by_name(session, "Test Snowflake")
            if existing:
                logger.info("Test data source already exists, skipping creation")
            else:
                session.add(test_source)
                await session.commit()
                logger.info("✓ Data source creation successful")
            
            # Retrieve the data source
            source = await get_data_source_by_name(session, "Test Snowflake")
            if source:
                logger.info(f"✓ Data source retrieval successful: {source.name}")
            else:
                logger.error("✗ Data source retrieval failed")
                return False
            
        logger.info("Data source operations test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Data source operations test failed: {str(e)}")
        return False
    finally:
        await close_database()

async def run_tests():
    """Run all MCP integration tests."""
    logger.info("Starting MCP integration tests...")
    
    # Run the tests
    db_test = await test_database_connection()
    model_test = await test_model_operations()
    source_test = await test_data_source_operations()
    
    # Summarize results
    logger.info("MCP Integration Test Summary:")
    logger.info(f"Database Connection: {'✓ PASSED' if db_test else '✗ FAILED'}")
    logger.info(f"Model Operations: {'✓ PASSED' if model_test else '✗ FAILED'}")
    logger.info(f"Data Source Operations: {'✓ PASSED' if source_test else '✗ FAILED'}")
    
    # Overall result
    overall = db_test and model_test and source_test
    logger.info(f"Overall: {'✓ PASSED' if overall else '✗ FAILED'}")
    
    return overall

if __name__ == "__main__":
    # Run the integration tests
    result = asyncio.run(run_tests())
    
    # Exit with code 0 if all tests pass, 1 otherwise
    sys.exit(0 if result else 1) 