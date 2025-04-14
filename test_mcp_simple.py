#!/usr/bin/env python3
"""
Simple MCP Integration Test

This script verifies the basic functionality of our MCP integration
without requiring a full database setup.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mcp_test")

def test_file_operations():
    """Test basic file operations for our MCP storage backends."""
    logger.info("Testing file operations...")
    
    # Create test directories for each storage backend
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    backends = ["local", "s3", "azure", "snowflake"]
    for backend in backends:
        backend_dir = storage_dir / backend
        backend_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Created directory for {backend} backend")
    
    # Create a sample model file
    model_file = storage_dir / "models.json"
    sample_model = {
        "id": "test-model-1",
        "name": "Test Model",
        "description": "A test model for MCP integration",
        "backend": "openai",
        "version": "1.0.0",
        "api_base": "https://api.openai.com/v1",
        "configuration": {"model_name": "gpt-4"},
        "is_active": True,
        "metrics": {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "average_latency": 0.0
        }
    }
    
    with open(model_file, "w") as f:
        json.dump([sample_model], f, indent=2)
    logger.info(f"✓ Created sample model file: {model_file}")
    
    # Create a sample data source file
    source_file = storage_dir / "data_sources.json"
    sample_source = {
        "id": "test-snowflake-1",
        "name": "Test Snowflake",
        "type": "snowflake",
        "description": "A test Snowflake data source",
        "configuration": {
            "account": "your-account",
            "user": "your-user",
            "password": "your-password",
            "warehouse": "your-warehouse",
            "database": "your-database",
            "schema": "your-schema"
        },
        "is_active": True
    }
    
    with open(source_file, "w") as f:
        json.dump([sample_source], f, indent=2)
    logger.info(f"✓ Created sample data source file: {source_file}")
    
    return True

def test_mcp_module_structure():
    """Test that our MCP module structure is correct."""
    logger.info("Testing MCP module structure...")
    
    # Check that key directories exist
    directories = [
        "mcp",
        "mcp/database",
        "mcp/server",
        "mcp/storage"
    ]
    
    for directory in directories:
        if os.path.isdir(directory):
            logger.info(f"✓ Directory exists: {directory}")
        else:
            logger.error(f"✗ Directory missing: {directory}")
            return False
    
    # Check that key files exist
    files = [
        "mcp/__init__.py",
        "mcp/database/__init__.py",
        "mcp/database/config.py",
        "mcp/database/models.py",
        "mcp/server/__init__.py",
        "mcp/server/config.py",
        "mcp/storage/__init__.py",
        "mcp/storage/config.py",
        "mcp/types.py"
    ]
    
    for file in files:
        if os.path.isfile(file):
            logger.info(f"✓ File exists: {file}")
        else:
            logger.error(f"✗ File missing: {file}")
            return False
    
    return True

def test_app_module_structure():
    """Test that our app module structure is correct."""
    logger.info("Testing app module structure...")
    
    # Check that key directories exist
    directories = [
        "app/core",
        "app/api"
    ]
    
    for directory in directories:
        if os.path.isdir(directory):
            logger.info(f"✓ Directory exists: {directory}")
        else:
            logger.error(f"✗ Directory missing: {directory}")
            return False
    
    # Check that key files exist
    files = [
        "app/core/database.py",
        "app/core/models.py",
        "app/core/config.py",
        "app/core/logger.py",
        "app/core/model_client.py",
        "app/api/mcp_server.py"
    ]
    
    for file in files:
        if os.path.isfile(file):
            logger.info(f"✓ File exists: {file}")
        else:
            logger.error(f"✗ File missing: {file}")
            return False
    
    return True

def test_env_file():
    """Test that our .env file has the required settings."""
    logger.info("Testing .env file...")
    
    if not os.path.isfile(".env"):
        logger.error("✗ .env file missing")
        return False
    
    # Read .env file
    with open(".env", "r") as f:
        env_content = f.read()
    
    # Check for required settings
    required_settings = [
        "APP_NAME",
        "DATABASE_URL",
        "SECRET_KEY",
        "STORAGE_BACKEND"
    ]
    
    for setting in required_settings:
        if setting in env_content:
            logger.info(f"✓ Setting exists: {setting}")
        else:
            logger.error(f"✗ Setting missing: {setting}")
            return False
    
    return True

def run_tests():
    """Run all tests."""
    logger.info("Starting MCP integration tests...")
    
    # Run the tests
    file_test = test_file_operations()
    mcp_module_test = test_mcp_module_structure()
    app_module_test = test_app_module_structure()
    env_test = test_env_file()
    
    # Summarize results
    logger.info("MCP Integration Test Summary:")
    logger.info(f"File Operations: {'✓ PASSED' if file_test else '✗ FAILED'}")
    logger.info(f"MCP Module Structure: {'✓ PASSED' if mcp_module_test else '✗ FAILED'}")
    logger.info(f"App Module Structure: {'✓ PASSED' if app_module_test else '✗ FAILED'}")
    logger.info(f"Environment Settings: {'✓ PASSED' if env_test else '✗ FAILED'}")
    
    # Overall result
    overall = file_test and mcp_module_test and app_module_test and env_test
    logger.info(f"Overall: {'✓ PASSED' if overall else '✗ FAILED'}")
    
    return overall

if __name__ == "__main__":
    # Run the tests
    result = run_tests()
    
    # Exit with code 0 if all tests pass, 1 otherwise
    sys.exit(0 if result else 1) 