import asyncio
import logging
from fastmcp import Context
from app.mcp_server import mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockContext:
    """Mock context for testing"""
    def __init__(self):
        class MockRequestContext:
            def __init__(self):
                self.lifespan_context = None
                self.context = {}
        self.request_context = MockRequestContext()

async def direct_tool_test():
    """Test calling clientview tools directly"""
    # Get all tools
    tools = await mcp.get_tools()
    
    # Check if get_top_clients is available
    if 'get_top_clients' in tools:
        logger.info("Found get_top_clients tool, calling it directly...")
        
        # Create a mock context
        ctx = MockContext()
        
        # Call the tool function directly
        tool_fn = tools['get_top_clients'].fn
        result = await tool_fn(ctx, sorting="top", currency="USD")
        
        logger.info(f"Tool result preview: {result[:300]}...")
        print("\nFull tool result:")
        print(result)
        
        return True
    else:
        logger.error("get_top_clients tool not found!")
        return False

if __name__ == "__main__":
    asyncio.run(direct_tool_test()) 