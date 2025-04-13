from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncpg
from datetime import datetime
import uuid
import os

from app.core.metadata_manager import MetadataManager
from app.core.query_mapper import QueryMapper
from app.core.data_router import DataSourceRouter

app = FastAPI(title="MCP Query Interface")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/mcp"
)

class QueryRequest(BaseModel):
    query: str
    api_key: str

class QueryResponse(BaseModel):
    query_id: str
    result: Dict[str, Any]
    execution_time: float
    source: str
    row_count: int

async def get_db():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/api/query", response_model=QueryResponse)
async def execute_query(
    query_request: QueryRequest,
    conn = Depends(get_db)
):
    try:
        # Initialize components
        metadata_manager = MetadataManager(DATABASE_URL)
        query_mapper = QueryMapper()
        data_router = DataSourceRouter({
            "postgres": {
                "enabled": True,
                "connection": DATABASE_URL
            }
        })

        # Start timing
        start_time = datetime.now()

        # Map query to SQL
        sql_query, query_type, params, target_source = query_mapper.map_query_with_source(
            query_request.query
        )

        if not sql_query:
            raise HTTPException(
                status_code=400,
                detail="Could not understand the query"
            )

        # Execute query
        result = await data_router.route_query(
            sql_query,
            query_request.api_key,
            {target_source: {"read": True}}  # Simplified permissions for example
        )

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Log query
        query_id = str(uuid.uuid4())
        await metadata_manager.log_query({
            "id": query_id,
            "api_key": query_request.api_key,
            "natural_query": query_request.query,
            "sql_query": sql_query,
            "data_source_id": target_source.value,
            "execution_time": execution_time,
            "row_count": len(result.get("data", [])),
            "status": result["status"],
            "error": result.get("error"),
            "response": result
        })

        return QueryResponse(
            query_id=query_id,
            result=result,
            execution_time=execution_time,
            source=target_source.value,
            row_count=len(result.get("data", []))
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/sources")
async def list_sources(conn = Depends(get_db)):
    """List available data sources and their tables"""
    metadata_manager = MetadataManager(DATABASE_URL)
    sources = await metadata_manager.get_all_sources()
    return sources

@app.get("/api/query_history")
async def get_query_history(
    api_key: str,
    limit: int = 100,
    conn = Depends(get_db)
):
    """Get query history for an API key"""
    metadata_manager = MetadataManager(DATABASE_URL)
    history = await metadata_manager.get_query_history(api_key, limit)
    return history 