from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from .models import ModelRecord, ModelUsageLog
from .database import get_db
from .logger import logger
import json
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

class RoutingMetrics:
    def __init__(self):
        self.latency: float = 0.0
        self.success_rate: float = 1.0
        self.load: int = 0
        self.last_updated = datetime.utcnow()
        self.query_types: Dict[str, int] = defaultdict(int)

class DataSourceHealth:
    def __init__(self):
        self.is_available: bool = True
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        self.last_checked = datetime.utcnow()
        self.recovery_attempts: int = 0
        self.last_recovery: Optional[datetime] = None

class QueryAnalyzer:
    PATTERNS = {
        'timeseries': [
            r'time\s+series',
            r'timestamp',
            r'date\s+range',
            r'interval',
            r'trend'
        ],
        'document': [
            r'text',
            r'document',
            r'content',
            r'search',
            r'match'
        ],
        'relational': [
            r'join',
            r'table',
            r'where',
            r'group\s+by',
            r'order\s+by'
        ]
    }
    
    def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        scores = {}
        query = query.lower()
        
        # Pattern matching
        for qtype, patterns in self.PATTERNS.items():
            scores[qtype] = sum(1 for pattern in patterns if pattern in query) / len(patterns)
        
        # Context-based adjustment
        if context and (recent_queries := context.get('recent_queries', [])):
            recent_types = defaultdict(int)
            for recent in recent_queries[-5:]:  # Last 5 queries
                if source := recent.get('source'):
                    recent_types[source] += 1
            
            # Boost scores based on recent usage
            for qtype in scores:
                if qtype in recent_types:
                    scores[qtype] *= 1 + 0.2 * recent_types[qtype]
        
        return scores

class IntelligentRouter:
    def __init__(self, db: Session):
        self.db = db
        self.routing_cache: Dict[str, Dict[str, Any]] = {}
        self.performance_stats: Dict[str, RoutingMetrics] = {}
        self.source_health: Dict[str, DataSourceHealth] = {}
        self.query_analyzer = QueryAnalyzer()
        self.fallback_routes: Dict[str, List[str]] = {}
        self.source_groups: Dict[str, List[str]] = {}
        
    async def update_metrics(
        self,
        model_id: str,
        source_id: str,
        execution_time: float,
        success: bool,
        query_type: Optional[str] = None
    ):
        """Update performance metrics with query type tracking"""
        key = f"{model_id}:{source_id}"
        if key not in self.performance_stats:
            self.performance_stats[key] = RoutingMetrics()
        
        metrics = self.performance_stats[key]
        metrics.latency = 0.7 * metrics.latency + 0.3 * execution_time
        metrics.success_rate = 0.7 * metrics.success_rate + 0.3 * float(success)
        metrics.last_updated = datetime.utcnow()
        
        if query_type:
            metrics.query_types[query_type] += 1
        metrics.load = sum(metrics.query_types.values())

    async def check_source_health(self, source_id: str, force_check: bool = False) -> bool:
        """Enhanced health check with recovery logic"""
        if source_id not in self.source_health:
            self.source_health[source_id] = DataSourceHealth()
        
        health = self.source_health[source_id]
        now = datetime.utcnow()
        
        if not health.is_available and (
            force_check or 
            health.last_recovery is None or 
            now - health.last_recovery > timedelta(minutes=5 * (health.recovery_attempts + 1))
        ):
            try:
                health.is_available = True
                health.error_count = 0
                health.last_error = None
                health.last_recovery = now
                logger.info(f"Data source {source_id} recovered successfully")
            except Exception as e:
                health.recovery_attempts += 1
                health.last_error = str(e)
                logger.warning(f"Recovery attempt failed for {source_id}: {str(e)}")
        
        health.last_checked = now
        return health.is_available and health.error_count < 3

    async def get_model_permissions(self, model_id: str) -> List[str]:
        """Get list of data sources a model has access to"""
        if model := self.db.query(ModelRecord).filter(ModelRecord.id == model_id).first():
            return model.config.get('allowed_sources', [])
        return []

    async def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Enhanced query analysis with context awareness"""
        return self.query_analyzer.analyze(query, context)

    def calculate_source_score(
        self,
        source_id: str,
        model_id: str,
        query_scores: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted source score with detailed breakdown"""
        key = f"{model_id}:{source_id}"
        metrics = self.performance_stats.get(key, RoutingMetrics())
        
        weights = {
            'performance': 0.25,
            'success_rate': 0.25,
            'query_match': 0.3,
            'load': 0.1,
            'context': 0.1
        }
        
        scores = {
            'performance': 1.0 / (1.0 + metrics.latency),
            'success_rate': metrics.success_rate,
            'query_match': query_scores.get(source_id, 0.0),
            'load': 1.0 / (1.0 + metrics.load),
            'context': context.get('preferred_sources', {}).get(source_id, 0.0) if context else 0.0
        }
        
        total_score = sum(weights[k] * scores[k] for k in weights)
        return total_score, scores

    async def route_request(
        self,
        model_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhanced request routing with fallback logic"""
        if not (allowed_sources := await self.get_model_permissions(model_id)):
            raise PermissionError(f"Model {model_id} has no accessible data sources")
        
        query_scores = await self.analyze_query(query, context)
        source_scores: Dict[str, Tuple[float, Dict[str, float]]] = {}
        
        # Try primary sources
        for source_id in allowed_sources:
            if await self.check_source_health(source_id):
                score, components = self.calculate_source_score(
                    source_id, model_id, query_scores, context
                )
                source_scores[source_id] = (score, components)
        
        # Try fallback sources if needed
        if not source_scores:
            for source_id in self.fallback_routes.get(model_id, []):
                if source_id in allowed_sources and await self.check_source_health(source_id, force_check=True):
                    score, components = self.calculate_source_score(
                        source_id, model_id, query_scores, context
                    )
                    source_scores[source_id] = (score, components)
        
        if not source_scores:
            raise RuntimeError("No healthy data sources available")
        
        best_source, (score, components) = max(
            source_scores.items(),
            key=lambda x: x[1][0]
        )
        
        self.routing_cache[model_id] = {
            'source': best_source,
            'timestamp': datetime.utcnow(),
            'context': context,
            'scores': components
        }
        
        logger.info(
            f"Routed model {model_id} to source {best_source} "
            f"with score {score:.2f} "
            f"(components: {json.dumps(components)})"
        )
        
        return best_source

    async def report_error(self, source_id: str, error: str):
        """Enhanced error reporting with group impact"""
        if source_id not in self.source_health:
            self.source_health[source_id] = DataSourceHealth()
        
        health = self.source_health[source_id]
        health.error_count += 1
        health.last_error = error
        health.last_checked = datetime.utcnow()
        
        if health.error_count >= 3:
            health.is_available = False
            logger.warning(f"Data source {source_id} marked as unavailable due to errors")
            
            for group, sources in self.source_groups.items():
                if source_id in sources and not any(
                    self.source_health[s].is_available for s in sources
                ):
                    logger.error(f"All sources in group {group} are unavailable!")

    async def reset_source_health(self, source_id: str):
        """Reset health status with recovery tracking"""
        if source_id in self.source_health:
            health = DataSourceHealth()
            health.last_recovery = datetime.utcnow()
            self.source_health[source_id] = health
            logger.info(f"Health status reset for data source {source_id}")

class PermissionError(Exception):
    pass 