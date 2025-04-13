from typing import Dict, Any, Optional, List
import hashlib
import json
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass
import sqlparse
from sqlparse.sql import Token, TokenList
import logging

@dataclass
class QueryStats:
    execution_time: float
    row_count: int
    cache_hit: bool
    optimizations_applied: List[str]
    timestamp: datetime

class QueryCache:
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0)
        )
        self.default_ttl = redis_config.get("default_ttl", 300)  # 5 minutes

    def _generate_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from query and parameters"""
        key_data = {
            "query": query,
            "params": params
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def get(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached query results"""
        key = self._generate_key(query, params)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set(self, query: str, params: Dict[str, Any], result: Dict[str, Any], ttl: Optional[int] = None):
        """Cache query results"""
        key = self._generate_key(query, params)
        self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(result)
        )

class QueryOptimizer:
    def __init__(self):
        self.optimization_patterns = [
            (r"SELECT \* FROM", "Avoid SELECT *", "Specify needed columns explicitly"),
            (r"GROUP BY.*HAVING", "Consider indexing HAVING clause columns", "Add indexes for HAVING clause columns"),
            (r"OR.*OR", "Multiple OR conditions", "Consider UNION or table redesign"),
            (r"NOT IN.*SELECT", "NOT IN subquery", "Consider EXISTS or JOIN"),
            (r"SELECT.*SELECT.*SELECT", "Multiple nested subqueries", "Consider CTEs or JOIN operations")
        ]

    def analyze_query(self, query: str) -> List[str]:
        """Analyze query for potential optimizations"""
        suggestions = []
        parsed = sqlparse.parse(query)[0]
        
        # Check for basic patterns
        for pattern, issue, suggestion in self.optimization_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                suggestions.append(f"{issue}: {suggestion}")

        # Analyze query structure
        self._analyze_structure(parsed, suggestions)
        
        return suggestions

    def _analyze_structure(self, parsed_query: TokenList, suggestions: List[str]):
        """Analyze query structure for optimization opportunities"""
        # Check for DISTINCT usage
        if any(token.value.upper() == 'DISTINCT' for token in parsed_query.tokens):
            suggestions.append("DISTINCT usage: Consider if DISTINCT is really necessary")

        # Check for implicit type conversions
        self._check_implicit_conversions(parsed_query, suggestions)

        # Check for proper indexing hints
        self._check_indexing(parsed_query, suggestions)

    def _check_implicit_conversions(self, parsed_query: TokenList, suggestions: List[str]):
        """Check for potential implicit type conversions"""
        comparison_tokens = [token for token in parsed_query.tokens if isinstance(token, Token)]
        for token in comparison_tokens:
            if token.ttype == Token.Operator.Comparison:
                suggestions.append("Check for implicit type conversions in comparisons")

    def _check_indexing(self, parsed_query: TokenList, suggestions: List[str]):
        """Check for proper indexing opportunities"""
        where_clause = False
        order_by_clause = False
        
        for token in parsed_query.tokens:
            if token.is_keyword:
                if token.value.upper() == 'WHERE':
                    where_clause = True
                elif token.value.upper() == 'ORDER BY':
                    order_by_clause = True

        if where_clause and order_by_clause:
            suggestions.append("Consider composite index for WHERE and ORDER BY columns")

class QueryAnalytics:
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 1)
        )
        self.stats_ttl = redis_config.get("stats_ttl", 86400 * 7)  # 7 days

    async def record_query(self, query: str, stats: QueryStats):
        """Record query execution statistics"""
        key = f"query_stats:{datetime.utcnow().strftime('%Y-%m-%d')}"
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        stats_data = {
            "execution_time": stats.execution_time,
            "row_count": stats.row_count,
            "cache_hit": stats.cache_hit,
            "optimizations_applied": stats.optimizations_applied,
            "timestamp": stats.timestamp.isoformat()
        }
        
        # Store in Redis
        self.redis.hset(key, query_hash, json.dumps(stats_data))
        self.redis.expire(key, self.stats_ttl)

    async def get_daily_stats(self, date: datetime) -> Dict[str, Any]:
        """Get query statistics for a specific day"""
        key = f"query_stats:{date.strftime('%Y-%m-%d')}"
        stats = self.redis.hgetall(key)
        
        if not stats:
            return {}
            
        return {
            k.decode(): json.loads(v.decode())
            for k, v in stats.items()
        }

    async def get_slow_queries(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get queries that took longer than threshold seconds"""
        slow_queries = []
        
        # Get stats for the last 7 days
        for i in range(7):
            date = datetime.utcnow() - timedelta(days=i)
            daily_stats = await self.get_daily_stats(date)
            
            for query_hash, stats in daily_stats.items():
                if stats["execution_time"] > threshold:
                    slow_queries.append({
                        "query_hash": query_hash,
                        **stats
                    })
                    
        return sorted(slow_queries, key=lambda x: x["execution_time"], reverse=True) 