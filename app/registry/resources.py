"""
Resource Registry Module

This module provides a registry for managing and accessing resources in the MCP server.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel

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

class ResourceRegistry:
    """Registry for external resources and API integrations"""
    def __init__(self):
        self._resources: Dict[str, ResourceDefinition] = {}
    
    def register(self, resource: ResourceDefinition):
        """Register a new resource"""
        self._resources[resource.full_name] = resource
    
    def get_resource(self, name: str, namespace: Optional[str] = None) -> ResourceDefinition:
        """
        Get a resource by name, optionally with namespace
        
        Args:
            name: Resource name
            namespace: Optional namespace. If not provided, will look for fully qualified name
                       or search in the default namespace.
        """
        # Check if the name already contains a namespace
        if ":" in name:
            if name not in self._resources:
                raise KeyError(f"Resource {name} not found")
            return self._resources[name]
        
        # Use provided namespace or default
        full_name = f"{namespace or 'default'}:{name}"
        if full_name not in self._resources:
            raise KeyError(f"Resource {full_name} not found")
        return self._resources[full_name]
    
    def list_resources(self, namespace: Optional[str] = None) -> Dict[str, ResourceDefinition]:
        """
        List all registered resources
        
        Args:
            namespace: Optional namespace to filter by
        """
        if namespace:
            return {
                name: resource 
                for name, resource in self._resources.items() 
                if name.startswith(f"{namespace}:")
            }
        return self._resources.copy()
    
    def list_namespaces(self) -> Dict[str, int]:
        """List all namespaces and resource count"""
        namespaces = {}
        for name in self._resources.keys():
            namespace = name.split(":", 1)[0]
            namespaces[namespace] = namespaces.get(namespace, 0) + 1
        return namespaces

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