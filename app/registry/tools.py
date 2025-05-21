"""
Tool Registry Module

This module provides a registry for managing and accessing tools in the MCP server.
"""

from typing import Dict, Any, Callable, Optional, List, Coroutine
from pydantic import BaseModel
import importlib
import pkgutil
import logging
import asyncio
from uuid import uuid4

from app.config import config

logger = logging.getLogger(__name__)

class ToolDefinition(BaseModel):
    name: str
    description: str
    handler: Callable
    namespace: str = "default"
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    @property
    def full_name(self) -> str:
        """Get the fully qualified name (namespace:name)"""
        return f"{self.namespace}:{self.name}"

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(self, tool: ToolDefinition):
        """Register a new tool"""
        self._tools[tool.full_name] = tool
    
    def get_tool(self, name: str, namespace: Optional[str] = None) -> ToolDefinition:
        """
        Get a tool by name, optionally with namespace
        
        Args:
            name: Tool name
            namespace: Optional namespace. If not provided, will look for fully qualified name
                       or search in the default namespace.
        """
        # Check if the name already contains a namespace
        if ":" in name:
            if name not in self._tools:
                raise KeyError(f"Tool {name} not found")
            return self._tools[name]
        
        # Use provided namespace or default
        full_name = f"{namespace or 'default'}:{name}"
        if full_name not in self._tools:
            raise KeyError(f"Tool {full_name} not found")
        return self._tools[full_name]
    
    def list_tools(self, namespace: Optional[str] = None) -> Dict[str, str]:
        """
        List all registered tools and their descriptions
        
        Args:
            namespace: Optional namespace to filter by
        """
        if namespace:
            return {
                name.split(":", 1)[1]: tool.description 
                for name, tool in self._tools.items() 
                if name.startswith(f"{namespace}:")
            }
        return {name: tool.description for name, tool in self._tools.items()}
    
    def list_namespaces(self) -> Dict[str, int]:
        """List all namespaces and tool count"""
        namespaces = {}
        for name in self._tools.keys():
            namespace = name.split(":", 1)[0]
            namespaces[namespace] = namespaces.get(namespace, 0) + 1
        return namespaces

# Create global registry instance
registry = ToolRegistry()


def _run_async(coro: Coroutine) -> None:
    """Helper to run a coroutine in the background if possible."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


async def _index_tool(tool: "ToolDefinition") -> None:
    """Add or update a tool document in the Compass index."""
    if not all([config.COMPASS_API_URL, config.COMPASS_BEARER_TOKEN, config.COMPASS_INDEX_NAME]):
        return

    try:
        from cohere_compass.clients.compass import CompassClient
        from cohere_compass.models.documents import (
            CompassDocument,
            CompassDocumentMetadata,
            CompassDocumentChunk,
        )

        client = CompassClient(
            index_url=config.COMPASS_API_URL,
            bearer_token=config.COMPASS_BEARER_TOKEN,
        )

        doc_id = f"tool-{tool.namespace}-{tool.name}"

        doc = CompassDocument(
            metadata=CompassDocumentMetadata(document_id=doc_id, filename=f"{tool.name}.md"),
            content={
                "text": f"{tool.full_name} - {tool.description}",
                "type": "tool",
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
            },
            chunks=[
                CompassDocumentChunk(
                    chunk_id=str(uuid4()),
                    sort_id="0",
                    document_id=doc_id,
                    parent_document_id=doc_id,
                    content={"text": f"{tool.full_name} - {tool.description}"},
                )
            ],
        )

        await asyncio.to_thread(
            client.insert_doc,
            index_name=config.COMPASS_INDEX_NAME,
            doc=doc,
        )

        logger.info("Indexed tool %s in Compass", tool.full_name)

    except Exception as exc:
        logger.error("Failed to index tool %s: %s", tool.full_name, exc)

def register_tool(name: str, description: str, namespace: str = "default", 
                  input_schema: Dict[str, Any] = None, output_schema: Dict[str, Any] = None,
                  metadata: Dict[str, Any] = None):
    """
    Decorator for registering tool handlers
    
    Example:
        @register_tool("search", "Search for documents", namespace="docs")
        async def search_docs(query: str) -> Dict[str, Any]:
            # tool implementation
            pass
    """
    def decorator(func):
        tool = ToolDefinition(
            name=name,
            description=description,
            handler=func,
            namespace=namespace,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            metadata=metadata or {}
        )
        registry.register(tool)
        try:
            _run_async(_index_tool(tool))
        except Exception as exc:
            logger.error("Error indexing tool %s: %s", tool.full_name, exc)
        return func
    return decorator 

def get_registered_tools(namespace: Optional[str] = None) -> List[ToolDefinition]:
    """
    Get a list of all registered tools, optionally filtered by namespace.
    
    Args:
        namespace: Optional namespace to filter tools by
        
    Returns:
        List of ToolDefinition objects for all registered tools
    """
    if namespace:
        return [
            tool for name, tool in registry._tools.items()
            if name.startswith(f"{namespace}:")
        ]
    return list(registry._tools.values())

def get_tool_by_name(name: str, namespace: Optional[str] = None) -> Optional[ToolDefinition]:
    """
    Get a tool by its name, optionally with namespace.
    
    Args:
        name: Tool name
        namespace: Optional namespace
        
    Returns:
        ToolDefinition if found, None otherwise
    """
    try:
        return registry.get_tool(name, namespace)
    except KeyError:
        return None


def autodiscover_tools(mcp, package: str = "app.tools") -> None:
    """Auto-import modules in ``package`` and register tools."""
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:
        logger.warning(f"Failed to import tools package {package}: {exc}")
        return

    for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if ispkg:
            continue
        module_name = f"{package}.{name}"
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "register_tools"):
                try:
                    mod.register_tools(mcp)
                except Exception as exc:
                    logger.warning(f"register_tools failed for {module_name}: {exc}")
        except Exception as exc:
            logger.warning(f"Error importing tool module {module_name}: {exc}")

# Export commonly used functions and classes
__all__ = [
    'ToolDefinition',
    'ToolRegistry',
    'register_tool',
    'get_registered_tools',
    'get_tool_by_name',
    'registry',
    'autodiscover_tools'
]
