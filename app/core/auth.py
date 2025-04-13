from datetime import datetime, timedelta
from typing import Dict, Optional, Set, List
import secrets
import logging
from pydantic import BaseModel
import sqlite3
import os
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    key_id: str
    key: str
    owner: str
    created_at: datetime
    expires_at: datetime
    permissions: List[str]
    is_active: bool
    last_used: Optional[datetime]
    usage_count: int

    @classmethod
    def from_db_row(cls, row: tuple) -> 'APIKey':
        return cls(
            key_id=row[0],
            key=row[1],
            owner=row[2],
            created_at=datetime.fromisoformat(row[3]),
            expires_at=datetime.fromisoformat(row[4]),
            permissions=json.loads(row[5]),
            is_active=bool(row[6]),
            last_used=datetime.fromisoformat(row[7]) if row[7] else None,
            usage_count=row[8]
        )

class APIKeyManager:
    """Manage API keys and authentication"""
    
    def __init__(self, db_path: str = "data/keys.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key TEXT UNIQUE NOT NULL,
                    owner TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    last_used TEXT,
                    usage_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    def generate_key(self, owner: str, expires_at: datetime,
                    permissions: List[str]) -> APIKey:
        """Generate a new API key."""
        key_id = secrets.token_urlsafe(16)
        key = secrets.token_urlsafe(32)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_keys (
                    key_id, key, owner, created_at, expires_at,
                    permissions, is_active, usage_count
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 0)
            """, (
                key_id, key, owner,
                datetime.utcnow().isoformat(),
                expires_at.isoformat(),
                json.dumps(permissions)
            ))
            conn.commit()

        return APIKey(
            key_id=key_id,
            key=key,
            owner=owner,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            permissions=permissions,
            is_active=True,
            last_used=None,
            usage_count=0
        )
    
    def validate_key(self, key: str, required_permissions: Optional[List[str]] = None) -> bool:
        """Validate an API key and update its usage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT key_id, permissions, is_active, expires_at
                FROM api_keys WHERE key = ?
            """, (key,))
            row = cursor.fetchone()
            
            if not row:
                return False
                
            key_id, permissions_json, is_active, expires_at = row
            permissions = json.loads(permissions_json)
            expires_at = datetime.fromisoformat(expires_at)
            
            if not is_active or expires_at < datetime.utcnow():
                return False
                
            if required_permissions:
                if not all(p in permissions for p in required_permissions):
                    return False

            # Update usage statistics
            conn.execute("""
                UPDATE api_keys
                SET last_used = ?, usage_count = usage_count + 1
                WHERE key_id = ?
            """, (datetime.utcnow().isoformat(), key_id))
            conn.commit()
            
            return True
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE api_keys SET is_active = 0
                WHERE key_id = ? AND is_active = 1
            """, (key_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_key_info(self, key_id: str) -> Optional[APIKey]:
        """Get information about a specific API key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM api_keys WHERE key_id = ?", (key_id,))
            row = cursor.fetchone()
            return APIKey.from_db_row(row) if row else None
    
    def list_keys(self, owner: Optional[str] = None) -> List[APIKey]:
        """List all API keys, optionally filtered by owner."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM api_keys"
            params = []
            
            if owner:
                query += " WHERE owner = ?"
                params.append(owner)
                
            cursor = conn.execute(query, params)
            return [APIKey.from_db_row(row) for row in cursor.fetchall()]

# Global API key manager instance
api_key_manager = APIKeyManager()

# Example usage:
"""
# Generate a key for a new user
key = api_key_manager.generate_key(
    owner="john.doe@company.com",
    expires_at=datetime.utcnow() + timedelta(days=90),
    permissions=["read", "write"]
)

# Share the key with the user
print(f"Your API key: {key.key}")

# Later, validate the key
if api_key_manager.validate_key(user_provided_key):
    # Allow access
    pass
else:
    # Deny access
    pass
""" 