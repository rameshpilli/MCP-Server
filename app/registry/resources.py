"""
Resource Registry Module

This module provides a registry for managing and accessing resources in the MCP server.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel

from .base import BaseRegistry

class ResourceDefinition(BaseModel):
    """Definition of an external resource or API integration"""
    name: str
    description: str
    type: str = "unknown"
    category: str = "general"
    metadata: Dict[str, Any] = {}
    namespace: str = "default"
    
    @property
    def full_name(self) -> str:
        """Get the fully qualified name (namespace:name)"""
        return f"{self.namespace}:{self.name}"

class ResourceRegistry(BaseRegistry[ResourceDefinition]):
    """Registry for external resources and API integrations."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def _resources(self) -> Dict[str, ResourceDefinition]:
        return self._items

    @_resources.setter
    def _resources(self, value: Dict[str, ResourceDefinition]) -> None:  # pragma: no cover - legacy
        self._items = value

    # Aliases for backward compatibility
    get_resource = BaseRegistry.get
    list_resources = BaseRegistry.list_items

# Create global registry instance
registry = ResourceRegistry()

def register_resource(name: str, description: str, type: str = "unknown", 
                     category: str = "general", namespace: str = "default",
                     metadata: Dict[str, Any] = None):
    """
    Decorator for registering resources
    
    Example:
        @register_resource("api_key", "API key for external service", type="credential")
        def get_api_key() -> str:
            return "secret-key"
    """
    def decorator(func):
        resource = ResourceDefinition(
            name=name,
            description=description,
            type=type,
            category=category,
            namespace=namespace,
            metadata=metadata or {}
        )
        registry.register(resource)
        return func
    return decorator

def get_registered_resources(namespace: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get a dictionary of all registered resources, optionally filtered by namespace.
    
    Args:
        namespace: Optional namespace to filter resources by
        
    Returns:
        Dictionary mapping resource names to their metadata
    """
    resources = registry.list_resources(namespace)
    return {
        name: {
            "description": resource.description,
            "type": resource.type,
            "category": resource.category,
            "namespace": resource.namespace,
            "metadata": resource.metadata
        }
        for name, resource in resources.items()
    }

def get_resource_by_name(name: str, namespace: Optional[str] = None) -> Optional[ResourceDefinition]:
    """
    Get a resource by its name, optionally with namespace.
    
    Args:
        name: Resource name
        namespace: Optional namespace
        
    Returns:
        ResourceDefinition if found, None otherwise
    """
    try:
        return registry.get_resource(name, namespace)
    except KeyError:
        return None

# Export commonly used functions and classes
__all__ = [
    'ResourceDefinition',
    'ResourceRegistry',
    'register_resource',
    'get_registered_resources',
    'get_resource_by_name',
    'registry'
] 
