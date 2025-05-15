import asyncio
import logging
import os 
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.mcp_server import mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_tools():
    """Check what tools are registered with the MCP server"""
    tools = await mcp.get_tools()
    
    logger.info(f"Found {len(tools)} registered tools:")
    
    # Group tools by namespace/prefix
    tool_groups = {}
    for name in sorted(tools.keys()):
        prefix = name.split(':')[0] if ':' in name else 'default'
        if prefix not in tool_groups:
            tool_groups[prefix] = []
        tool_groups[prefix].append(name)
    
    # Print tools by group
    for prefix, tools_list in sorted(tool_groups.items()):
        logger.info(f"\n{prefix} namespace:")
        for tool in sorted(tools_list):
            logger.info(f"  - {tool}")
    
    # Check specifically for clientview tools
    clientview_tools = [name for name in tools.keys() if 'client' in name.lower()]
    logger.info(f"\nClientView tools: {clientview_tools}")
    
    # Print detailed info for the first clientview tool if found
    if clientview_tools:
        tool_name = clientview_tools[0]
        logger.info(f"\nDetails for tool '{tool_name}':")
        tool = tools[tool_name]
        for attr_name in dir(tool):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(tool, attr_name)
                    if not callable(attr_value):  # Skip methods
                        logger.info(f"  {attr_name}: {attr_value}")
                except Exception:
                    pass
    
if __name__ == "__main__":
    asyncio.run(check_tools()) 