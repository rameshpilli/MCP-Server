# This test is for MCP_Bridge end-to-end flow.
# tests/test_end_to_end.py
import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from app.mcp_bridge import MCPBridge
from app.client import mcp_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_e2e")

async def test_end_to_end():
    """Test the end-to-end flow from query to result"""

    # Define test queries
    test_queries = [
        "Show me the top clients in USD",
        "What are the revenue trends for BlackRock?",
        "Show client value by product for JP Morgan",
        "Give me client data by quarter for this year",
        "Who are the top gainers in the USA region?",
        "Show me Focus40 clients with declining revenue in CAD"
    ]

    # Process each query
    for query in test_queries:
        logger.info(f"\n===== Testing query: '{query}' =====")

        # Process through the main client flow
        session_id = f"test_session_{hash(query) % 1000}"
        result = await mcp_client.process_message(query, session_id)

        # Log the results
        logger.info(f"Response: {result['response'][:100]}...")
        logger.info(f"Tools executed: {result['tools_executed']}")
        if 'context' in result:
            logger.info(f"Context info: {result.get('context', {})}")
        logger.info(f"Processing time: {result.get('processing_time_ms', 0)} ms")

        # Verify parameters were correctly passed
        # This would require modifications to your tool functions to log the parameters they receive

        logger.info("=" * 50)

        # Wait between requests to avoid rate limits
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_end_to_end())


# TODO - ERROR WITH get_client_value_by_time --> tool when processing "What are the revenue trends for BlackRock?
#  Error with register_tool --> llm seems to be extracting client_cdrid as a parameter, but the tool function doesn't accept that parameter"
# Some --> returning No data available for few queries maybe because ---> API doens't have data for tbose sepcific parameters.
# There might be a mismatch between what parameters the API expects and what' we're passing.

# Caching working --> parameters are being re-used without needing to call the LLM again for similar queries
