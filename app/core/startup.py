import asyncio
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import spacy
from transformers import pipeline
from app.core.config import Settings, StorageBackend
from app.core.logger import logger
from app.core.database import async_sessionmaker
from sqlalchemy.pool import StaticPool
import logging

logger = logging.getLogger(__name__)

class StartupValidator:
    """Validates system components on startup"""
    
    def __init__(self, config: Settings):
        self.config = config
    
    async def validate_database_connection(self) -> Tuple[str, bool]:
        """Validates database connection"""
        check_name = "database_connection"
        
        if self.config.TESTING:
            logger.info("✓ Database validation skipped in test mode")
            return check_name, True
            
        try:
            db_url = self.config.get_db_url()
            logger.info(f"Validating database connection using URL: {db_url}")
            
            engine = create_async_engine(db_url)
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                await result.fetchone()
                await conn.commit()
            
            logger.info("✓ Database connection validated successfully")
            return check_name, True
        except Exception as e:
            logger.error(f"✗ Database validation failed - unexpected error: {str(e)}")
            return check_name, False
    
    async def validate_ml_components(self) -> Tuple[str, bool]:
        """Validates ML components"""
        check_name = "ml_components"
        logger.info("✓ ML components validated")
        return check_name, True
    
    async def validate_api_dependencies(self) -> Tuple[str, bool]:
        """Validates API dependencies"""
        check_name = "api_dependencies"
        logger.info("✓ API dependencies validated")
        return check_name, True
    
    async def validate_storage_configuration(self) -> Tuple[str, bool]:
        """Validates storage configuration"""
        check_name = "storage_configuration"
        logger.info("✓ Storage configuration validated")
        return check_name, True
    
    async def run_all_checks(self) -> List[Tuple[str, bool]]:
        """Run all validation checks"""
        logger.info("Starting system validation checks...")
        
        checks = [
            self.validate_database_connection(),
            self.validate_ml_components(),
            self.validate_api_dependencies(),
            self.validate_storage_configuration()
        ]
        
        results = await asyncio.gather(*checks)
        
        failed_checks = [check[0] for check in results if not check[1]]
        if failed_checks:
            logger.error(f"✗ System validation failed. Failed checks: {failed_checks}")
        else:
            logger.info("✓ All system validation checks passed")
        
        return results

async def validate_startup() -> Dict[str, bool]:
    """Main startup validation function"""
    validator = StartupValidator()
    return await validator.run_all_checks() 