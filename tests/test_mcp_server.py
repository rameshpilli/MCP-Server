import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_server")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server
from app.mcp_server import mcp

async def test_server():
    """Test the MCP server functionality"""
    logger.info("Testing MCP server...")
    
    # Get all registered tools
    tools = await mcp.get_tools()
    logger.info(f"Found {len(tools)} registered tools")
    
    # Group tools by namespace
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
    
    # Test a specific tool if available
    test_tool_name = 'get_top_clients'
    if test_tool_name in tools:
        logger.info(f"\nTesting {test_tool_name} tool...")
        try:
            # Create a mock context
            class MockContext:
                def __init__(self):
                    class MockRequestContext:
                        def __init__(self):
                            self.lifespan_context = None
                            self.context = {}
                    self.request_context = MockRequestContext()
            
            ctx = MockContext()
            tool_fn = tools[test_tool_name].fn
            result = await tool_fn(ctx, sorting="top", currency="USD")
            
            logger.info(f"Tool result preview: {result[:300]}...")
            logger.info("Tool test completed successfully")
        except Exception as e:
            logger.error(f"Error testing tool: {e}")
    else:
        logger.warning(f"Test tool '{test_tool_name}' not found")

if __name__ == "__main__":
    logger.info("Starting MCP server tests...")
    asyncio.run(test_server())
    logger.info("Tests completed.") 