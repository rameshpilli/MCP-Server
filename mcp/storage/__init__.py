"""
MCP Storage Module

This module provides storage backends for MCP.
"""

from .config import StorageConfig

def create_storage_backend(config: StorageConfig):
    """Create a storage backend based on configuration."""
    # This is a mock implementation that will be replaced with real implementations
    if config.type == "local":
        from .local import LocalStorage
        return LocalStorage(config)
    elif config.type == "azure":
        from .azure import AzureStorage
        return AzureStorage(config)
    elif config.type == "s3":
        from .s3 import S3Storage
        return S3Storage(config)
    elif config.type == "snowflake":
        from .snowflake import SnowflakeStorage
        return SnowflakeStorage(config)
    else:
        raise ValueError(f"Unsupported storage type: {config.type}") 