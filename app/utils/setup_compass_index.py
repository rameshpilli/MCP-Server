"""
Utility for setting up Cohere Compass index for tool and resource discovery
"""

import logging
import json
from pathlib import Path
from typing import Dict
import httpx
from app.config import Config
from app.registry.tools import get_registered_tools, ToolDefinition
from app.registry.resources import get_registered_resources
from app.utils.parameter_extractor import _call_llm

logger = logging.getLogger(__name__)

# Path for caching tool classification results
CACHE_FILE = Path(__file__).resolve().parent.parent / "output" / "tool_classification_cache.json"


def _load_cache() -> Dict[str, Dict[str, str]]:
    """Load classification cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict[str, str]]) -> None:
    """Persist classification cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


async def classify_tool(tool: ToolDefinition, cache: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Classify a tool's intent and category using an LLM."""
    if tool.full_name in cache:
        return cache[tool.full_name]

    system_prompt = (
        "You label tools with an intent and a short category. "
        "Respond with JSON using keys 'intent' and 'category'."
    )
    user_prompt = f"Tool name: {tool.full_name}\nDescription: {tool.description}"

    try:
        response = await _call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(response[start : end + 1])
        else:
            data = json.loads(response)
        intent = data.get("intent", "general")
        category = data.get("category", "general")
    except Exception as e:
        logger.warning(f"LLM classification failed for {tool.full_name}: {e}")
        intent = "general"
        category = "general"

    cache[tool.full_name] = {"intent": intent, "category": category}
    _save_cache(cache)
    return cache[tool.full_name]

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

        cache: Dict[str, Dict[str, str]] = _load_cache()
        
        # Prepare documents for indexing
        documents = []
        
        # Add tools
        for tool in tools:
            classification = await classify_tool(tool, cache)
            doc = {
                "id": f"tool:{tool.full_name}",
                "type": "tool",
                "content": {
                    "name": tool.full_name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "intent": classification.get("intent", "general"),
                    "category": classification.get("category", "general"),
                    "namespace": tool.namespace,
                    "metadata": tool.metadata,
                },
                "text": f"Tool: {tool.full_name} - {tool.description}",
                "metadata": {
                    "type": "tool",
                    "namespace": tool.namespace,
                    "description": tool.description,
                },
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