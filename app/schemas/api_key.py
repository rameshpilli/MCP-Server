from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class APIKeyCreate(BaseModel):
    owner: str
    expiry_days: Optional[int] = None
    permissions: List[str] = []
    rate_limit: str = "5/minute"

class APIKeyResponse(BaseModel):
    key: str
    owner: str
    expires_at: datetime
    permissions: List[str]
    rate_limit: str
    
    class Config:
        from_attributes = True 