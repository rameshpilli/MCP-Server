"""
Short Term Memory

This module provides a short-term memory implementation for the MCP server.
"""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
import redis

from app.config import Config

logger = logging.getLogger("mcp_server.memory")


class ShortTermMemory:
    """
    Short-term memory implementation using Redis
    """

    def __init__(self, session_id: str = None):
        """
        Initialize the short-term memory

        Args:
            session_id: Optional session ID for the memory
        """
        config = Config()
        self.session_id = session_id or f"session-{int(time.time())}"
        self.max_messages = 10  # Maximum number of messages to store

        # Try to connect to Redis if configured
        self.redis_client = None
        try:
            if config.REDIS_HOST:
                redis_url = config.get_redis_url()
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
            else:
                logger.info("No Redis configuration found, using in-memory storage")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")

        # Fallback to in-memory if Redis is not available
        self.in_memory_store = {}

    def _get_key(self, key: str) -> str:
        """Get namespaced key."""
        return f"memory:{self.session_id}:{key}"
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in memory.
        
        Args:
            key: Key to store value under
            value: Value to store
            ttl: Optional time-to-live in seconds
        """
        try:
            if self.redis_client:
                self.redis_client.setex(
                    self._get_key(key),
                    ttl or 3600,
                    json.dumps(value)
                )
            else:
                self.in_memory_store[self._get_key(key)] = {
                    "value": value,
                    "expires": time.time() + (ttl or 3600)
                }
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Stored value or None if not found/expired
        """
        try:
            if self.redis_client:
                value = self.redis_client.get(self._get_key(key))
                return json.loads(value) if value else None
            else:
                entry = self.in_memory_store.get(self._get_key(key))
                if entry and time.time() < entry["expires"]:
                    return entry["value"]
                return None
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return None
    
    async def delete(self, key: str) -> None:
        """Delete a value from memory.
        
        Args:
            key: Key to delete
        """
        try:
            if self.redis_client:
                self.redis_client.delete(self._get_key(key))
            else:
                self.in_memory_store.pop(self._get_key(key), None)
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise
    
    async def clear(self) -> None:
        """Clear all memory for this session."""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(f"memory:{self.session_id}:*")
                if keys:
                    self.redis_client.delete(*keys)
            else:
                self.in_memory_store.clear()
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise

    async def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a message to the memory
        
        Args:
            role: The role of the sender (user, assistant, system)
            content: The message content
            metadata: Optional metadata for the message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # Store in Redis if available
            if self.redis_client:
                # Get current messages
                messages_key = self._get_key("messages")
                messages = []
                
                try:
                    messages_data = self.redis_client.get(messages_key)
                    if messages_data:
                        messages = json.loads(messages_data)
                except Exception as e:
                    logger.warning(f"Error reading from Redis: {e}")
                
                # Add new message
                messages.append(message)
                
                # Trim messages if needed
                if len(messages) > self.max_messages:
                    messages = messages[-self.max_messages:]
                
                # Save back to Redis
                self.redis_client.set(messages_key, json.dumps(messages))
                
                # Set expiration (24 hours)
                self.redis_client.expire(messages_key, 86400)
            else:
                # In-memory storage
                if self._get_key("messages") not in self.in_memory_store:
                    self.in_memory_store[self._get_key("messages")] = []
                
                self.in_memory_store[self._get_key("messages")].append(message)
                
                # Trim messages if needed
                if len(self.in_memory_store[self._get_key("messages")]) > self.max_messages:
                    self.in_memory_store[self._get_key("messages")] = self.in_memory_store[self._get_key("messages")][-self.max_messages:]
            
            return True
        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")
            return False
    
    async def get_messages(self, count: int = None) -> List[Dict[str, Any]]:
        """
        Get messages from memory
        
        Args:
            count: Number of most recent messages to retrieve (all if None)
            
        Returns:
            List of messages
        """
        try:
            # Get from Redis if available
            if self.redis_client:
                messages_key = self._get_key("messages")
                messages_data = self.redis_client.get(messages_key)
                
                if messages_data:
                    messages = json.loads(messages_data)
                else:
                    messages = []
            else:
                # In-memory storage
                messages = self.in_memory_store.get(self._get_key("messages"), [])
            
            # Return requested number of messages
            if count is not None:
                return messages[-count:]
            else:
                return messages
        except Exception as e:
            logger.error(f"Error getting messages from memory: {e}")
            return []
    
    async def get_context(self) -> Dict[str, Any]:
        """
        Get conversation context for use in LLM requests
        
        Returns:
            Dictionary containing conversation context
        """
        messages = await self.get_messages()
        
        return {
            "session_id": self.session_id,
            "message_count": len(messages),
            "messages": messages
        } 