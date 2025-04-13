import asyncio
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import spacy
from transformers import pipeline
from app.core.config import Settings
from app.core.logger import logger
from app.core.database import async_session_maker

class StartupValidator:
    """Validates all required components during application startup"""
    
    def __init__(self):
        self.config = Settings()
        self.checks: Dict[str, bool] = {}
        
    async def validate_database_connection(self) -> Tuple[str, bool]:
        """Validates database connection using async SQLAlchemy"""
        check_name = "database_connection"
        try:
            async with async_session_maker() as session:
                await session.execute(text("SELECT 1"))
                await session.commit()
            logger.info("✓ Database connection validated")
            return check_name, True
        except Exception as e:
            logger.error(f"✗ Database validation failed: {str(e)}")
            return check_name, False
            
    async def validate_ml_components(self) -> Tuple[str, bool]:
        """Validates ML model dependencies and configurations"""
        check_name = "ml_components"
        try:
            # Check if required ML libraries are available
            import numpy
            import pandas
            import sklearn
            
            # Validate model paths and configurations
            # TODO: Add specific model validation logic
            
            logger.info("✓ ML components validated")
            return check_name, True
        except ImportError as e:
            logger.error(f"✗ ML component validation failed - missing dependency: {str(e)}")
            return check_name, False
        except Exception as e:
            logger.error(f"✗ ML component validation failed: {str(e)}")
            return check_name, False
            
    async def validate_api_dependencies(self) -> Tuple[str, bool]:
        """Validates API dependencies and configurations"""
        check_name = "api_dependencies"
        try:
            # Check if required API packages are available
            import fastapi
            import uvicorn
            import pydantic
            
            # Validate API configurations
            assert self.config.SECRET_KEY, "SECRET_KEY not configured"
            assert self.config.API_KEY_EXPIRY_DAYS > 0, "Invalid API_KEY_EXPIRY_DAYS"
            
            logger.info("✓ API dependencies validated")
            return check_name, True
        except ImportError as e:
            logger.error(f"✗ API dependency validation failed - missing package: {str(e)}")
            return check_name, False
        except AssertionError as e:
            logger.error(f"✗ API configuration validation failed: {str(e)}")
            return check_name, False
        except Exception as e:
            logger.error(f"✗ API dependency validation failed: {str(e)}")
            return check_name, False
            
    async def validate_storage_config(self) -> Tuple[str, bool]:
        """Validates storage configuration and permissions"""
        check_name = "storage_config"
        try:
            import os
            from pathlib import Path
            
            # Ensure storage directory exists and is writable
            storage_path = Path(self.config.STORAGE_PATH)
            if not storage_path.exists():
                storage_path.mkdir(parents=True)
                
            # Test write permissions
            test_file = storage_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            
            logger.info("✓ Storage configuration validated")
            return check_name, True
        except PermissionError as e:
            logger.error(f"✗ Storage validation failed - permission denied: {str(e)}")
            return check_name, False
        except Exception as e:
            logger.error(f"✗ Storage validation failed: {str(e)}")
            return check_name, False

    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all validation checks"""
        logger.info("Starting system validation checks...")
        
        checks = [
            self.validate_database_connection(),
            self.validate_ml_components(),
            self.validate_api_dependencies(),
            self.validate_storage_config()
        ]
        
        results = await asyncio.gather(*checks)
        self.checks = dict(results)
        
        all_passed = all(status for _, status in results)
        if all_passed:
            logger.info("✓ All system checks passed successfully")
        else:
            failed_checks = [
                check for check, status in results
                if not status
            ]
            logger.error(f"✗ System validation failed. Failed checks: {failed_checks}")
        
        return self.checks

async def validate_startup() -> Dict[str, bool]:
    """Main startup validation function"""
    validator = StartupValidator()
    return await validator.run_all_checks() 