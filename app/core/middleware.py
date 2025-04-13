from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, Dict, List, Tuple
from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, select
from .models import APIKey, ModelUsageLog, UsageQuota
from .database import async_sessionmaker, get_db
from .security import APIKeyManager, create_api_key_hash
from .logger import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response, JSONResponse
import time
from collections import defaultdict
import asyncio
import logging
import re
from fastapi import FastAPI
from .config import get_settings

logger = logging.getLogger(__name__)

class RateLimitExceeded(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=429, detail=detail)

class RateLimiter:
    """Tracks API requests and enforces rate limits."""

    def __init__(self):
        settings = get_settings()
        # Store requests as {api_key: {timestamp: count}}
        self.requests = {}
        self._cleanup_interval = settings.RATE_LIMIT_CLEANUP_INTERVAL
        self._last_cleanup = time.time()
        self._default_limit = settings.RATE_LIMIT_DEFAULT_LIMIT
        self._default_window = settings.RATE_LIMIT_DEFAULT_WINDOW

    def parse_rate_limit(self, rate_limit: Optional[str] = None) -> Tuple[int, str]:
        """Parse rate limit string in format 'X/minute' or 'X/hour'.
        
        Args:
            rate_limit: Rate limit string in format 'X/minute' or 'X/hour'
            
        Returns:
            Tuple of (limit, unit) where unit is either 'minute' or 'hour'
        """
        if not rate_limit:
            return (self._default_limit, self._default_window)  # Use default from settings

        try:
            count, unit = rate_limit.split("/")
            count = int(count)
            unit = unit.lower()
            if unit not in ("minute", "hour"):
                raise ValueError
            return (count, unit)
        except (ValueError, AttributeError):
            return (self._default_limit, self._default_window)  # Default on parsing error

    def _cleanup_old_entries(self):
        """Remove entries older than the cleanup interval."""
        current_time = time.time()
        if current_time - self._last_cleanup >= self._cleanup_interval:
            for api_key in list(self.requests.keys()):
                # Remove timestamps older than cleanup interval
                self.requests[api_key] = {
                    ts: count for ts, count in self.requests[api_key].items()
                    if current_time - ts < self._cleanup_interval
                }
                # Remove empty API keys
                if not self.requests[api_key]:
                    del self.requests[api_key]
            self._last_cleanup = current_time

    def is_rate_limited(self, api_key: str, rate_limit: Optional[str] = None) -> bool:
        """Check if an API key has exceeded its rate limit.
        
        Args:
            api_key: The API key to check
            rate_limit: Rate limit string in format 'X/minute' or 'X/hour'
            
        Returns:
            True if rate limited, False otherwise
        """
        self._cleanup_old_entries()
        
        # Initialize request tracking for new API keys
        if api_key not in self.requests:
            self.requests[api_key] = {}

        current_time = time.time()
        limit, unit = self.parse_rate_limit(rate_limit)
        window = 60 if unit == "minute" else 3600

        # Count requests in current window
        window_start = current_time - window
        total_requests = sum(
            count for ts, count in self.requests[api_key].items()
            if ts >= window_start
        )

        # Check if adding this request would exceed the limit
        return total_requests >= limit

    def add_request(self, api_key: str, rate_limit: Optional[str] = None):
        """Add a request to the counter.
        
        Args:
            api_key: The API key to add a request for
            rate_limit: Rate limit string in format 'X/minute' or 'X/hour'
        """
        current_time = time.time()
        if api_key not in self.requests:
            self.requests[api_key] = {}
        self.requests[api_key][current_time] = 1

    def get_remaining_requests(self, api_key: str, rate_limit: Optional[str] = None) -> Tuple[int, int]:
        """Get the number of remaining requests for an API key.
        
        Args:
            api_key: The API key to check
            rate_limit: Rate limit string in format 'X/minute' or 'X/hour'
            
        Returns:
            Tuple of (limit, remaining_requests)
        """
        limit, unit = self.parse_rate_limit(rate_limit)
        window = 60 if unit == "minute" else 3600
        
        if api_key not in self.requests:
            return limit, limit  # No requests yet

        current_time = time.time()
        window_start = current_time - window
        used_requests = sum(
            count for ts, count in self.requests[api_key].items()
            if ts >= window_start
        )
        
        remaining = max(0, limit - used_requests)
        return limit, remaining

    def get_window_reset_time(self, api_key: str, rate_limit: Optional[str] = None) -> int:
        """Get the number of seconds until the current rate limit window resets.
        
        Args:
            api_key: The API key to check
            rate_limit: Rate limit string in format 'X/minute' or 'X/hour'
            
        Returns:
            Seconds until window reset
        """
        _, unit = self.parse_rate_limit(rate_limit)
        window = 60 if unit == "minute" else 3600
        
        if api_key not in self.requests:
            return 0
            
        current_time = time.time()
        oldest_request = min(self.requests[api_key].keys(), default=current_time)
        time_passed = current_time - oldest_request
        
        return max(0, int(window - time_passed))

async def get_api_key_from_header(request: Request) -> Optional[str]:
    return request.headers.get("X-API-Key")

async def get_api_key_info(db: AsyncSession, api_key: str) -> Optional[APIKey]:
    """Get API key information from the database."""
    api_key_manager = APIKeyManager()
    api_key_record = await api_key_manager.validate_key(db, api_key)
    return api_key_record

async def check_rate_limit(db: AsyncSession, api_key_record: APIKey) -> None:
    settings = get_settings()
    
    # Parse rate limit string (e.g., "100/day" or "1000/month")
    if not api_key_record.rate_limit:
        return
        
    limit, period = api_key_record.rate_limit.split('/')
    limit = int(limit)
    
    # Calculate time window
    now = datetime.now(timezone.utc)
    if period == 'minute':
        window_start = now - timedelta(minutes=1)
    elif period == 'hour':
        window_start = now - timedelta(hours=1)
    elif period == 'day':
        window_start = now - timedelta(days=1)
    elif period == 'month':
        window_start = now - timedelta(days=30)
    else:
        raise ValueError(f"Unsupported rate limit period: {period}")
    
    # Count recent requests
    result = await db.execute(
        select(ModelUsageLog).where(
            and_(
                ModelUsageLog.api_key_id == api_key_record.id,
                ModelUsageLog.timestamp > window_start
            )
        )
    )
    request_count = len(result.all())
    
    if request_count >= limit:
        raise RateLimitExceeded(
            f"Rate limit exceeded: {limit} requests per {period}. "
            f"Current usage: {request_count} requests"
        )

async def check_quota(db: AsyncSession, api_key_record: APIKey) -> None:
    settings = get_settings()
    if settings.TESTING:
        # In test mode, don't check quotas
        return

    result = await db.execute(
        select(UsageQuota).where(UsageQuota.api_key_id == api_key_record.id)
    )
    quota = result.scalar_one_or_none()
    
    if not quota:
        return  # No quota set
    
    # Check if quota needs to be reset
    now = datetime.utcnow()
    if quota.reset_frequency == 'daily' and (now - quota.last_reset).days >= 1:
        quota.last_reset = now
        await db.commit()
        return
    
    if quota.reset_frequency == 'monthly' and (now - quota.last_reset).days >= 30:
        quota.last_reset = now
        await db.commit()
        return
    
    # Check usage against quotas
    result = await db.execute(
        select(ModelUsageLog).where(
            and_(
                ModelUsageLog.api_key_id == api_key_record.id,
                ModelUsageLog.timestamp > quota.last_reset
            )
        )
    )
    usage_since_reset = result.scalars().all()
    
    total_requests = len(usage_since_reset)
    total_tokens = sum(log.tokens_used or 0 for log in usage_since_reset)
    total_cost = sum(log.cost or 0 for log in usage_since_reset)
    
    if quota.max_requests and total_requests >= quota.max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Request quota exceeded: {quota.max_requests} requests per {quota.reset_frequency}"
        )
    
    if quota.max_tokens and total_tokens >= quota.max_tokens:
        raise HTTPException(
            status_code=429,
            detail=f"Token quota exceeded: {quota.max_tokens} tokens per {quota.reset_frequency}"
        )
    
    if quota.max_cost and total_cost >= quota.max_cost:
        raise HTTPException(
            status_code=429,
            detail=f"Cost quota exceeded: ${quota.max_cost:.2f} per {quota.reset_frequency}"
        )

async def log_usage(
    db: AsyncSession,
    api_key_record: APIKey,
    model_id: int,
    request_type: str,
    tokens_used: int,
    execution_time: float,
    success: bool,
    error_message: Optional[str] = None,
    cost: Optional[float] = None,
    metadata: Optional[dict] = None
) -> None:
    settings = get_settings()
    if settings.TESTING:
        # In test mode, don't log usage
        return

    log_entry = ModelUsageLog(
        model_id=model_id,
        api_key_id=api_key_record.id,
        request_type=request_type,
        tokens_used=tokens_used,
        execution_time=execution_time,
        success=success,
        error_message=error_message,
        cost=cost,
        metadata=metadata
    )
    
    db.add(log_entry)
    await db.commit()
    
    logger.info(f"Usage logged for API key {api_key_record.key}: "
                f"{tokens_used} tokens, {execution_time:.2f}s, "
                f"success={success}, cost=${cost or 0:.4f}")

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and rate limiting."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        settings = get_settings()
        self.rate_limiter = RateLimiter()
        self.public_paths = settings.PUBLIC_PATHS
        self.no_rate_limit_paths = settings.NO_RATE_LIMIT_PATHS
        self.test_api_key = settings.TEST_API_KEY
        self.api_key_manager = APIKeyManager()

    async def validate_api_key(self, api_key: str, db: AsyncSession) -> Optional[APIKey]:
        """Validate API key and return API key info if valid."""
        try:
            settings = get_settings()
            if settings.TESTING:
                # In test mode, validate against the database directly
                hashed_key = create_api_key_hash(api_key)
                result = await db.execute(
                    select(APIKey).where(
                        APIKey.key == hashed_key,
                        APIKey.is_active == True
                    )
                )
                api_key_record = result.scalar_one_or_none()
                if api_key_record:
                    await db.refresh(api_key_record)
                return api_key_record
            return await self.api_key_manager.validate_key(db, api_key)
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()
        if request.url.path in self.public_paths:
            return await call_next(request)

        api_key = await get_api_key_from_header(request)
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"}
            )

        try:
            # Get database session using dependency injection
            db = await get_db().__anext__()
            
            api_key_record = await self.validate_api_key(api_key, db)
            if not api_key_record:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"}
                )

            if not settings.TESTING and api_key_record.expires_at and api_key_record.expires_at < datetime.now(timezone.utc):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key has expired"}
                )

            # Skip rate limit check for certain paths or in test mode
            skip_rate_limit = request.url.path in self.no_rate_limit_paths or settings.TESTING

            # Get rate limit info before checking limit
            limit, remaining = self.rate_limiter.get_remaining_requests(api_key, api_key_record.rate_limit)

            # Check rate limit for paths that require it
            if not skip_rate_limit:
                if self.rate_limiter.is_rate_limited(api_key, api_key_record.rate_limit):
                    reset_time = self.rate_limiter.get_window_reset_time(api_key)
                    response = JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded",
                            "reset_time": reset_time
                        }
                    )
                    response.headers["X-RateLimit-Limit"] = str(limit)
                    response.headers["X-RateLimit-Remaining"] = "0"
                    response.headers["X-RateLimit-Reset"] = str(reset_time)
                    return response

            # Store the API key record in the request state for later use
            request.state.api_key = api_key_record

            response = await call_next(request)
            
            # Add rate limit headers to response
            if not skip_rate_limit:
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(self.rate_limiter.get_window_reset_time(api_key))

            return response
        except Exception as e:
            logger.error(f"Error in API key middleware: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking API usage."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track API usage metrics."""
        start_time = time.time()
        
        # Get API key from header if available
        api_key = await get_api_key_from_header(request)
        
        # Record the request
        logger.info(f"API request - {request.method} {request.url.path} (API key: {'present' if api_key else 'missing'})")
        
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record the response
        logger.info(f"API response - {request.method} {request.url.path} completed in {duration:.2f}s with status {response.status_code}")
        
        return response 

class DatabaseSessionMiddleware(BaseHTTPMiddleware):
    """Middleware for managing database sessions."""
    
    async def dispatch(self, request: Request, call_next):
        """Create a new database session for each request."""
        settings = get_settings()
        
        # Create a new session
        if settings.TESTING:
            # In test mode, use the test engine
            from app.core.database import get_engine
            engine = get_engine()
            session = AsyncSession(bind=engine, expire_on_commit=False)
        else:
            from app.core.database import get_session_factory
            session_factory = get_session_factory()
            session = session_factory()

        # Store session in request state
        request.state.db = session
        
        try:
            response = await call_next(request)
            if not settings.TESTING:  # Don't commit in test mode
                await session.commit()
            return response
        except Exception as e:
            if not settings.TESTING:  # Don't rollback in test mode
                await session.rollback()
            raise e
        finally:
            await session.close() 