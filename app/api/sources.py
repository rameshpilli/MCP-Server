from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from app.core.dependencies import get_db
from app.core.config import Settings

router = APIRouter()
settings = Settings()

@router.get("/sources")
async def list_sources(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all available data sources."""
    # For now, return a static list of sources
    # In a real implementation, this would query a database
    return [
        {
            "id": "postgres_main",
            "name": "Main PostgreSQL Database",
            "type": "postgresql",
            "description": "Primary application database",
            "tables": ["users", "models", "usage_logs"]
        },
        {
            "id": "clickhouse_analytics",
            "name": "ClickHouse Analytics",
            "type": "clickhouse",
            "description": "Analytics data warehouse",
            "tables": ["events", "metrics", "aggregations"]
        }
    ]

@router.get("/sources/{source_id}")
async def get_source(source_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get details about a specific data source."""
    sources = {
        "postgres_main": {
            "id": "postgres_main",
            "name": "Main PostgreSQL Database",
            "type": "postgresql",
            "description": "Primary application database",
            "tables": ["users", "models", "usage_logs"],
            "connection": {
                "host": "localhost",
                "port": 5432,
                "database": "mcp"
            }
        },
        "clickhouse_analytics": {
            "id": "clickhouse_analytics",
            "name": "ClickHouse Analytics",
            "type": "clickhouse",
            "description": "Analytics data warehouse",
            "tables": ["events", "metrics", "aggregations"],
            "connection": {
                "host": "localhost",
                "port": 8123,
                "database": "analytics"
            }
        }
    }
    
    if source_id not in sources:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    
    return sources[source_id]

@router.get("/sources/{source_id}/tables")
async def list_tables(source_id: str, db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List tables available in a data source."""
    tables = {
        "postgres_main": [
            {
                "name": "users",
                "columns": ["id", "username", "email", "created_at"],
                "primary_key": "id"
            },
            {
                "name": "models",
                "columns": ["id", "name", "version", "created_at"],
                "primary_key": "id"
            },
            {
                "name": "usage_logs",
                "columns": ["id", "model_id", "timestamp", "request_type"],
                "primary_key": "id"
            }
        ],
        "clickhouse_analytics": [
            {
                "name": "events",
                "columns": ["timestamp", "event_type", "user_id", "data"],
                "primary_key": "timestamp, event_type"
            },
            {
                "name": "metrics",
                "columns": ["timestamp", "metric_name", "value"],
                "primary_key": "timestamp, metric_name"
            }
        ]
    }
    
    if source_id not in tables:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    
    return tables[source_id] 