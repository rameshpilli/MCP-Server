"""Enums used throughout the application."""
from enum import Enum

class DataSourceType(Enum):
    """Types of data sources supported by the application."""
    SNOWFLAKE = "snowflake"
    AZURE_STORAGE = "azure_storage"
    LOCAL_DB = "local_db"
    CUSTOM = "custom" 