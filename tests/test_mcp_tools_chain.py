import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_tools_chain")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server and bridge
from app.mcp_server import mcp
from app.mcp_bridge import MCPBridge

async def test_tool_chaining():
    """Test the MCP tool chaining functionality"""
    logger.info("Testing MCP tool chaining...")
    
    # Initialize the bridge
    bridge = MCPBridge()
    
    # Test queries that should trigger tool chaining
    test_queries = [
        "search for company policies and summarize the results",
        "read the q1_report document and analyze its sentiment",
        "get top clients and analyze their sentiment",
        "search for client feedback and summarize the key points"
    ]
    
    # Test each query
    for query in test_queries:
        logger.info(f"\n=== Testing chained query: '{query}' ===")
        
        try:
            # Get the routing result
            result = await bridge.route_request(query)
            
            # Check for chaining
            has_chaining = any('depends_on' in ep for ep in result['endpoints'])
            logger.info(f"Tool chaining detected: {has_chaining}")
            
            if has_chaining:
                # Display the chain
                logger.info("Tool chain:")
                for ep in result['endpoints']:
                    logger.info(f"  - {ep['name']} ({ep['type']})")
                    if ep.get('depends_on'):
                        logger.info(f"    Depends on: {ep['depends_on']}")
                    if ep.get('params'):
                        logger.info(f"    Parameters: {ep['params']}")
                
                # Try to execute the chain
                logger.info("\nExecuting tool chain...")
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
                    
                    # Execute each tool in the chain
                    chain_results = {}
                    for ep in result['endpoints']:
                        tool_name = ep['name']
                        if tool_name in mcp.tools:
                            # Get dependencies
                            deps = ep.get('depends_on', [])
                            if deps:
                                logger.info(f"Waiting for dependencies: {deps}")
                                # Wait for dependencies to complete
                                for dep in deps:
                                    if dep not in chain_results:
                                        logger.warning(f"Dependency {dep} not found in results")
                                        continue
                            
                            # Execute the tool
                            logger.info(f"Executing {tool_name}...")
                            tool_fn = mcp.tools[tool_name].fn
                            result = await tool_fn(ctx, **ep.get('params', {}))
                            chain_results[tool_name] = result
                            logger.info(f"Tool result preview: {result[:300]}...")
                        else:
                            logger.warning(f"Tool {tool_name} not found")
                    
                    logger.info("Tool chain execution completed")
                except Exception as e:
                    logger.error(f"Error executing tool chain: {e}")
            else:
                logger.info("No tool chaining detected for this query")
            
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
            logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting MCP tool chaining tests...")
    asyncio.run(test_tool_chaining())
    logger.info("Tests completed.") 