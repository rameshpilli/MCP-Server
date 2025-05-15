"""
Utility for setting up Cohere Compass index for tool and resource discovery
"""

import logging
from typing import Dict, Any, List, Optional
import httpx
from app.config import Config
from app.registry.tools import get_registered_tools, ToolDefinition
from app.registry.resources import get_registered_resources

logger = logging.getLogger(__name__)

def derive_intent_from_description(description: str) -> str:
    """Derive intent from tool description"""
    description_lower = description.lower()
    
    if any(kw in description_lower for kw in ["search", "find", "look for", "locate"]):
        return "search"
    elif any(kw in description_lower for kw in ["summarize", "summary", "brief"]):
        return "summarize"
    elif any(kw in description_lower for kw in ["analyze", "analysis", "insights"]):
        return "analyze"
    elif any(kw in description_lower for kw in ["report", "generate report", "create report"]):
        return "report"
    elif any(kw in description_lower for kw in ["document", "file", "read"]):
        return "document"
    else:
        return "general"

def derive_category_from_tool(name: str, tool: ToolDefinition) -> str:
    """Derive category from tool name and definition"""
    name_lower = name.lower()
    description_lower = tool.description.lower()
    
    # Check name first
    if any(kw in name_lower for kw in ["search", "find", "query"]):
        return "search"
    elif any(kw in name_lower for kw in ["summarize", "summary"]):
        return "summarization"
    elif any(kw in name_lower for kw in ["analyze", "analysis"]):
        return "analysis"
    elif any(kw in name_lower for kw in ["report", "generate"]):
        return "reporting"
    elif any(kw in name_lower for kw in ["document", "file"]):
        return "document"
    
    # Check description if name doesn't give a clear category
    if any(kw in description_lower for kw in ["search", "find", "query"]):
        return "search"
    elif any(kw in description_lower for kw in ["summarize", "summary"]):
        return "summarization"
    elif any(kw in description_lower for kw in ["analyze", "analysis"]):
        return "analysis"
    elif any(kw in description_lower for kw in ["report", "generate"]):
        return "reporting"
    elif any(kw in description_lower for kw in ["document", "file"]):
        return "document"
    
    return "general"

async def setup_compass_index():
    """Set up Cohere Compass index for tool and resource discovery"""
    try:
        config = Config()
        if not all([config.COMPASS_API_URL, config.COMPASS_BEARER_TOKEN, config.COMPASS_INDEX_NAME]):
            logger.error("Missing Compass configuration")
            return
        
        # Get all tools and resources
        tools = get_registered_tools()
        resources = get_registered_resources()
        
        # Prepare documents for indexing
        documents = []
        
        # Add tools
        for tool in tools:
            # Create document for tool
            doc = {
                "id": f"tool:{tool.full_name}",
                "type": "tool",
                "content": {
                    "name": tool.full_name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "intent": derive_intent_from_description(tool.description),
                    "category": derive_category_from_tool(tool.full_name, tool),
                    "namespace": tool.namespace,
                    "metadata": tool.metadata
                },
                "text": f"Tool: {tool.full_name} - {tool.description}",
                "metadata": {
                    "type": "tool",
                    "namespace": tool.namespace,
                    "description": tool.description
                }
            }
            documents.append(doc)
        
        # Add resources
        for resource_name, resource_info in resources.items():
            # Create document for resource
            doc = {
                "id": f"resource:{resource_name}",
                "type": "resource",
                "content": {
                    "name": resource_name,
                    "description": resource_info.get("description", ""),
                    "type": resource_info.get("type", "unknown"),
                    "category": resource_info.get("category", "general"),
                    "metadata": resource_info.get("metadata", {})
                },
                "text": f"Resource: {resource_name} - {resource_info.get('description', '')}",
                "metadata": {
                    "type": "resource",
                    "description": resource_info.get("description", "")
                }
            }
            documents.append(doc)
        
        # Upload documents to Compass
        async with httpx.AsyncClient() as client:
            # Create or update index
            response = await client.post(
                f"{config.COMPASS_API_URL}/indexes",
                headers={"Authorization": f"Bearer {config.COMPASS_BEARER_TOKEN}"},
                json={
                    "index_name": config.COMPASS_INDEX_NAME,
                    "documents": documents
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            logger.info(f"Successfully indexed {len(documents)} documents in Compass")
            
    except Exception as e:
        logger.error(f"Error setting up Compass index: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(setup_compass_index()) 