"""
Search Tools

This module contains tools for searching and retrieving information.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json

# Add parent directory to path to access agent modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))

from config import config
from app.registry.tools import register_tool

# Get logger
logger = logging.getLogger("mcp_agent.tools.search")

@register_tool(
    name="search_data",
    description="Search through local data files",
    namespace=config.AGENT_NAMESPACE,
    input_schema={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Maximum number of results to return"}
    }
)
async def search_data(ctx, query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search through data files in the data directory.

    Args:
        ctx: The context object
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    logger.info(f"Searching for '{query}' in data files")
    
    try:
        results = []
        data_dir = Path(config.DATA_DIR)
        
        # Search through all JSON files in the data directory
        for file_path in data_dir.glob("**/*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Simple search implementation - check if query is in any text field
                if _contains_query(data, query):
                    results.append({
                        "file": str(file_path.relative_to(data_dir)),
                        "title": data.get("title", "Untitled"),
                        "description": data.get("description", ""),
                        "path": str(file_path)
                    })
                    
                    if len(results) >= max_results:
                        break
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error in search_data: {e}")
        return {
            "error": str(e),
            "query": query,
            "results_count": 0,
            "results": []
        }

def _contains_query(data: Any, query: str) -> bool:
    """
    Check if the data contains the query string.
    
    Args:
        data: The data to search in
        query: The query string
    
    Returns:
        True if the query is found, False otherwise
    """
    query = query.lower()
    
    if isinstance(data, dict):
        for key, value in data.items():
            if _contains_query(key, query) or _contains_query(value, query):
                return True
    elif isinstance(data, list):
        for item in data:
            if _contains_query(item, query):
                return True
    elif isinstance(data, str):
        return query in data.lower()
    
    return False 