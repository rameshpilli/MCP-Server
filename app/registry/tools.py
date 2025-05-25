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

from .base import BaseRegistry

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

class ToolRegistry(BaseRegistry[ToolDefinition]):
    """Registry for tools."""

    def __init__(self) -> None:
        super().__init__()

    # Backwards compatible access to the underlying dict
    @property
    def _tools(self) -> Dict[str, ToolDefinition]:
        return self._items

    @_tools.setter
    def _tools(self, value: Dict[str, ToolDefinition]) -> None:  # pragma: no cover - legacy
        self._items = value

    # Aliases for previous method names
    get_tool = BaseRegistry.get
    list_tools = BaseRegistry.list_items

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
