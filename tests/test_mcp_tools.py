import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_tools")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server
from app.mcp_server import mcp

async def test_tools():
    """Test the MCP tools functionality"""
    logger.info("Testing MCP tools...")
    
    # Get all registered tools
    tools = await mcp.get_tools()
    logger.info(f"Found {len(tools)} registered tools")
    
    # Test each tool
    for name, tool in tools.items():
        logger.info(f"\n=== Testing tool: {name} ===")
        
        # Print tool details
        logger.info(f"Function: {tool.fn}")
        logger.info(f"Description: {getattr(tool, 'description', 'No description')}")
        
        # Try to get tool parameters if available
        try:
            params = getattr(tool, 'parameters', {})
            if params:
                logger.info("Parameters:")
                for param_name, param_info in params.items():
                    logger.info(f"  - {param_name}: {param_info.get('type', 'unknown')}")
                    if 'description' in param_info:
                        logger.info(f"    Description: {param_info['description']}")
        except Exception as e:
            logger.warning(f"Could not get parameters: {e}")
        
        # Test tool execution for specific tools
        if name == 'get_top_clients':
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
                result = await tool.fn(ctx, sorting="top", currency="USD")
                logger.info(f"Tool execution result preview: {result[:300]}...")
            except Exception as e:
                logger.error(f"Error executing tool: {e}")
        
        logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting MCP tools tests...")
    asyncio.run(test_tools())
    logger.info("Tests completed.") 