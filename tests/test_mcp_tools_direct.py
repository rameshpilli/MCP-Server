import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_tools_direct")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server
from app.mcp_server import mcp

async def test_direct_tools():
    """Test direct execution of MCP tools"""
    logger.info("Testing direct MCP tool execution...")
    
    # Get all registered tools
    tools = await mcp.get_tools()
    logger.info(f"Found {len(tools)} registered tools")
    
    # Create a mock context
    class MockContext:
        def __init__(self):
            class MockRequestContext:
                def __init__(self):
                    self.lifespan_context = None
                    self.context = {}
            self.request_context = MockRequestContext()
    
    ctx = MockContext()
    
    # Test specific tools directly
    test_tools = {
        'get_top_clients': {
            'params': {'sorting': 'top', 'currency': 'USD'},
            'description': 'Get top clients by value'
        },
        'search_documents': {
            'params': {'query': 'company policies'},
            'description': 'Search for documents'
        },
        'read_document': {
            'params': {'doc_name': 'company_policies'},
            'description': 'Read a specific document'
        }
    }
    
    # Test each tool
    for tool_name, tool_info in test_tools.items():
        if tool_name in tools:
            logger.info(f"\n=== Testing {tool_name} ===")
            logger.info(f"Description: {tool_info['description']}")
            
            try:
                # Get the tool function
                tool_fn = tools[tool_name].fn
                
                # Execute the tool
                logger.info(f"Executing with parameters: {tool_info['params']}")
                result = await tool_fn(ctx, **tool_info['params'])
                
                # Display the result
                logger.info(f"Tool result preview: {result[:300]}...")
                logger.info("Tool execution completed successfully")
            except Exception as e:
                logger.error(f"Error executing tool: {e}")
        else:
            logger.warning(f"Test tool '{tool_name}' not found")
        
        logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting direct MCP tools tests...")
    asyncio.run(test_direct_tools())
    logger.info("Tests completed.") 