from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from .query_mapper import QueryMapper
from .snowflake_connector import SnowflakeConnector
from .azure_storage import AzureStorageConnector
from .local_db import LocalDBConnector
from app.core.enums import DataSourceType

class DataSourcePermission:
    source_type: DataSourceType
    read: bool = False
    write: bool = False
    admin: bool = False

class DataSourceRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_mapper = QueryMapper()
        
        # Initialize connectors based on configuration
        self.connectors = {}
        if config.get("snowflake", {}).get("enabled", False):
            self.connectors[DataSourceType.SNOWFLAKE] = SnowflakeConnector(config["snowflake"])
        
        if config.get("azure_storage", {}).get("enabled", False):
            self.connectors[DataSourceType.AZURE_STORAGE] = AzureStorageConnector(config["azure_storage"])
            
        if config.get("local_db", {}).get("enabled", False):
            self.connectors[DataSourceType.LOCAL_DB] = LocalDBConnector(config["local_db"])

    async def route_query(
        self,
        query: str,
        api_key: str,
        token_permissions: Dict[DataSourceType, DataSourcePermission]
    ) -> Dict[str, Any]:
        """Route query to appropriate data source based on query type and permissions"""
        
        # Map the natural language query to SQL and determine data source
        sql_query, query_type, params, target_source = self.query_mapper.map_query_with_source(query)
        
        if not sql_query:
            return {
                "status": "error",
                "error": "Could not understand the query"
            }

        # Check if user has permission for the target data source
        if target_source not in token_permissions:
            return {
                "status": "error",
                "error": f"Your API key does not have access to {target_source.value}. "
                        f"Please contact xyz@gmail.com to request access."
            }

        permission = token_permissions[target_source]
        if not permission.read:
            return {
                "status": "error",
                "error": f"Your API key does not have read permission for {target_source.value}. "
                        f"Please contact xyz@gmail.com to upgrade your access."
            }

        # Check if we have a connector for the target source
        if target_source not in self.connectors:
            return {
                "status": "error",
                "error": f"Data source {target_source.value} is not currently enabled. "
                        f"Please contact system administrator."
            }

        # Execute query on the appropriate data source
        try:
            connector = self.connectors[target_source]
            result = await connector.execute_query(sql_query, params)
            
            return {
                "status": "success",
                "data": result,
                "source": target_source.value,
                "query_type": query_type
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "source": target_source.value
            }

    def get_available_sources(self, token_permissions: Dict[DataSourceType, DataSourcePermission]) -> List[Dict[str, Any]]:
        """Get list of data sources available to the user"""
        available_sources = []
        
        for source_type, permission in token_permissions.items():
            if source_type in self.connectors:
                available_sources.append({
                    "source": source_type.value,
                    "permissions": {
                        "read": permission.read,
                        "write": permission.write,
                        "admin": permission.admin
                    },
                    "status": "active"
                })
                
        return available_sources 