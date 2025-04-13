from typing import Optional, Dict, Any, List
from datetime import datetime, UTC
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.models import ModelUsageLog
from app.core.database import get_db
from app.core.logger import logger
import httpx
from pydantic import BaseModel
import asyncio

class ModelStats(BaseModel):
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_latency: float = 0.0
    last_query: Optional[datetime] = None
    sources_used: Dict[str, int] = {}

class ModelContext(BaseModel):
    recent_queries: List[Dict[str, Any]] = []
    preferred_sources: Dict[str, float] = {}
    last_updated: datetime = datetime.utcnow()
    max_history: int = 10

class QueryResponse(BaseModel):
    query_id: str
    result: Any
    execution_time: float
    source: str
    timestamp: datetime

class ModelClient:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str = "http://localhost:8000",
        context_ttl: int = 3600
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.context = ModelContext()
        self.stats = ModelStats()
        self.context_ttl = context_ttl
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}
        )
        self._lock = asyncio.Lock()

    async def register(self, config: Dict[str, Any]) -> bool:
        """Register model with the platform"""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/models/register",
                json={"model_id": self.model_id, "config": config}
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            raise RegistrationError(f"Failed to register model: {str(e)}")

    async def connect(self) -> bool:
        """Validate connection and load context"""
        try:
            response = await self._client.get(
                f"{self.base_url}/api/models/validate",
                params={"model_id": self.model_id}
            )
            response.raise_for_status()
            
            if context_data := response.json().get('context'):
                self.context = ModelContext(**context_data)
            
            return True
        except httpx.HTTPError as e:
            raise ConnectionError(f"Failed to connect: {str(e)}")

    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> QueryResponse:
        """Execute a query with context tracking"""
        async with self._lock:
            start_time = datetime.utcnow()
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/query",
                    json={
                        "model_id": self.model_id,
                        "query": query,
                        "context": {**self.context.dict(), **(context or {})}
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                await self._update_stats(True, execution_time, response.json())
                await self._update_context(query, response.json())
                
                return QueryResponse(**response.json())
                
            except httpx.HTTPError as e:
                await self._update_stats(False, (datetime.utcnow() - start_time).total_seconds())
                raise QueryError(f"Query failed: {str(e)}")

    async def _update_stats(self, success: bool, latency: float, response: Optional[Dict] = None):
        """Update model statistics"""
        try:
            usage_log = ModelUsageLog(
                model_id=self.model_id,
                query=response.get('query', ''),
                source_id=response.get('source'),
                execution_time=latency,
                status='success' if success else 'failed',
                model_metadata={
                    'context': self.context.dict(),
                    'response_metadata': response.get('metadata') if response else None
                }
            )
            self.db.add(usage_log)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to update stats: {str(e)}")

        self.stats.total_queries += 1
        self.stats.successful_queries += success
        self.stats.failed_queries += not success
        
        self.stats.avg_latency = self.stats.avg_latency * 0.7 + latency * 0.3 if self.stats.avg_latency else latency
        self.stats.last_query = datetime.utcnow()
        
        if response and (source := response.get('source')):
            self.stats.sources_used[source] = self.stats.sources_used.get(source, 0) + 1

    async def _update_context(self, query: str, response: Dict[str, Any]):
        """Update model context with query information"""
        self.context.recent_queries.append({
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'source': response.get('source'),
            'execution_time': response.get('execution_time')
        })
        
        if len(self.context.recent_queries) > self.context.max_history:
            self.context.recent_queries = self.context.recent_queries[-self.context.max_history:]
        
        if source := response.get('source'):
            current_score = self.context.preferred_sources.get(source, 0.5)
            success_score = float(response.get('status') == 'success')
            self.context.preferred_sources[source] = current_score * 0.7 + success_score * 0.3
        
        self.context.last_updated = datetime.utcnow()

    async def get_stats(self) -> ModelStats:
        """Get current model statistics"""
        return self.stats

    async def get_context(self) -> ModelContext:
        """Get current model context"""
        return self.context

    async def clear_context(self):
        """Reset model context"""
        self.context = ModelContext()

    async def close(self):
        """Close client connection and save context"""
        try:
            await self._client.post(
                f"{self.base_url}/api/models/{self.model_id}/context",
                json=self.context.dict()
            )
        finally:
            await self._client.aclose()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

class QueryError(Exception):
    pass

class RegistrationError(Exception):
    pass

class UsageError(Exception):
    pass 