import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_bridge")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCPBridge class
from app.mcp_bridge import MCPBridge

async def test_bridge():
    """Test the MCPBridge functionality"""
    logger.info("Initializing MCPBridge...")
    bridge = MCPBridge()
    
    # Test queries
    test_queries = [
        "search for information about company policies",
        "read the company_policies document",
        "analyze the sentiment of the following text: I love the new product features!",
        "summarize the q1_report document",
        # Add multi-step chain test
        "search for company policies and summarize the results",
        "read the q1_report document and analyze its sentiment"
    ]
    
    # Test each query
    for query in test_queries:
        logger.info(f"\n=== Testing query: '{query}' ===")
        
        # Call the router
        try:
            result = await bridge.route_request(query)
            
            # Display the routing results
            logger.info(f"Intent: {result['intent']}")
            logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
            if 'prompt_type' in result:
                logger.info(f"Prompt type: {result['prompt_type']}")
                
            # Display endpoints
            logger.info("Endpoints:")
            for ep in result['endpoints']:
                logger.info(f"  - {ep['name']} ({ep['type']})")
                if ep.get('params'):
                    logger.info(f"    Parameters: {ep['params']}")
                if ep.get('depends_on'):
                    logger.info(f"    Depends on: {ep['depends_on']}")
            
            # Check for chaining
            if any('depends_on' in ep for ep in result['endpoints']):
                logger.info("✓ Tool chaining detected!")
                
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
            logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting MCPBridge tests...")
    asyncio.run(test_bridge())
    logger.info("Tests completed.") 