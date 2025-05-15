import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_tools_search")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server
from app.mcp_server import mcp

async def test_search_tools():
    """Test the MCP search tools functionality"""
    logger.info("Testing MCP search tools...")
    
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
    
    # Test search queries
    test_queries = [
        "company policies",
        "client feedback",
        "quarterly report",
        "product roadmap",
        "security guidelines"
    ]
    
    # Test search tools
    search_tools = {
        'search_documents': {
            'description': 'Search through documents',
            'param_name': 'query'
        },
        'search_clientview': {
            'description': 'Search client view data',
            'param_name': 'query'
        }
    }
    
    # Test each search tool
    for tool_name, tool_info in search_tools.items():
        if tool_name in tools:
            logger.info(f"\n=== Testing {tool_name} ===")
            logger.info(f"Description: {tool_info['description']}")
            
            # Test each query
            for query in test_queries:
                logger.info(f"\nTesting query: '{query}'")
                
                try:
                    # Get the tool function
                    tool_fn = tools[tool_name].fn
                    
                    # Execute the search
                    params = {tool_info['param_name']: query}
                    logger.info(f"Executing with parameters: {params}")
                    result = await tool_fn(ctx, **params)
                    
                    # Display the result
                    logger.info(f"Search result preview: {result[:300]}...")
                    
                    # Try to parse and display structured results if available
                    try:
                        if isinstance(result, (list, tuple)):
                            logger.info(f"Found {len(result)} results")
                            for i, item in enumerate(result[:3], 1):  # Show first 3 results
                                logger.info(f"\nResult {i}:")
                                if isinstance(item, dict):
                                    for key, value in item.items():
                                        logger.info(f"  {key}: {value}")
                                else:
                                    logger.info(f"  {item}")
                        elif isinstance(result, dict):
                            logger.info("Result structure:")
                            for key, value in result.items():
                                logger.info(f"  {key}: {value}")
                    except Exception as e:
                        logger.warning(f"Could not parse structured results: {e}")
                    
                    logger.info("Search completed successfully")
                except Exception as e:
                    logger.error(f"Error executing search: {e}")
                
                logger.info("-" * 30)
        else:
            logger.warning(f"Search tool '{tool_name}' not found")
        
        logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting MCP search tools tests...")
    asyncio.run(test_search_tools())
    logger.info("Tests completed.") 