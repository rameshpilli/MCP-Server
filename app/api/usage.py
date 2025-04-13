from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from app.core.models import ModelUsageLog, APIKey
from app.core.dependencies import get_db

router = APIRouter()

class UsageStatistics(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: float
    average_latency: float
    success_rate: float
    requests_by_endpoint: dict
    usage_by_day: List[dict]

@router.get("/usage/stats")
async def get_usage_statistics(
    api_key: str = Query(..., description="API key to get statistics for"),
    days: int = Query(30, description="Number of days to look back"),
    db: Session = Depends(get_db)
) -> UsageStatistics:
    """Get detailed usage statistics for an API key"""
    
    # Validate API key
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not key_record:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Calculate time window
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get usage logs for the period
    logs = db.query(ModelUsageLog).filter(
        ModelUsageLog.api_key_id == key_record.id,
        ModelUsageLog.timestamp.between(start_date, end_date)
    ).all()
    
    if not logs:
        return UsageStatistics(
            total_requests=0,
            total_tokens=0,
            total_cost=0.0,
            average_latency=0.0,
            success_rate=0.0,
            requests_by_endpoint={},
            usage_by_day=[]
        )
    
    # Calculate basic statistics
    total_requests = len(logs)
    total_tokens = sum(log.tokens_used or 0 for log in logs)
    total_cost = sum(log.cost or 0 for log in logs)
    average_latency = sum(log.execution_time or 0 for log in logs) / total_requests
    successful_requests = sum(1 for log in logs if log.success)
    success_rate = (successful_requests / total_requests) * 100
    
    # Calculate requests by endpoint
    requests_by_endpoint = {}
    for log in logs:
        endpoint = log.endpoint or 'unknown'
        if endpoint not in requests_by_endpoint:
            requests_by_endpoint[endpoint] = {
                'count': 0,
                'tokens': 0,
                'cost': 0.0
            }
        requests_by_endpoint[endpoint]['count'] += 1
        requests_by_endpoint[endpoint]['tokens'] += log.tokens_used or 0
        requests_by_endpoint[endpoint]['cost'] += log.cost or 0
    
    # Calculate daily usage
    usage_by_day = []
    current_date = start_date
    while current_date <= end_date:
        day_logs = [log for log in logs if log.timestamp.date() == current_date.date()]
        usage_by_day.append({
            'date': current_date.date().isoformat(),
            'requests': len(day_logs),
            'tokens': sum(log.tokens_used or 0 for log in day_logs),
            'cost': sum(log.cost or 0 for log in day_logs),
            'success_rate': (sum(1 for log in day_logs if log.success) / len(day_logs) * 100) if day_logs else 0
        })
        current_date += timedelta(days=1)
    
    return UsageStatistics(
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_cost=total_cost,
        average_latency=average_latency,
        success_rate=success_rate,
        requests_by_endpoint=requests_by_endpoint,
        usage_by_day=usage_by_day
    )

@router.get("/usage/current")
async def get_current_usage(
    api_key: str = Query(..., description="API key to get current usage for"),
    db: Session = Depends(get_db)
):
    """Get current usage status and remaining quota for an API key"""
    
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not key_record:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Parse rate limit
    limit, period = key_record.rate_limit.split('/')
    limit = int(limit)
    
    # Calculate current window
    now = datetime.utcnow()
    if period == 'minute':
        window_start = now - timedelta(minutes=1)
    elif period == 'hour':
        window_start = now - timedelta(hours=1)
    elif period == 'day':
        window_start = now - timedelta(days=1)
    elif period == 'month':
        window_start = now - timedelta(days=30)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported rate limit period: {period}")
    
    # Get current usage
    current_usage = db.query(ModelUsageLog).filter(
        ModelUsageLog.api_key_id == key_record.id,
        ModelUsageLog.timestamp > window_start
    ).count()
    
    return {
        "rate_limit": key_record.rate_limit,
        "current_usage": current_usage,
        "remaining_requests": limit - current_usage,
        "reset_time": window_start + timedelta(
            minutes=1 if period == 'minute' else 0,
            hours=1 if period == 'hour' else 0,
            days=1 if period == 'day' else 30
        )
    } 