import asyncio
import logging
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_compass_index_setup")


# import app.main 
from app.mcp_server import mcp

# Add tool registration before indexing
from app.registry import tool_registry
# from app.tools.clientview_financials import register_tools as register_financial_tools
import app.tools.clientview_financials as clientview_financials

clientview_financials.register_tools(mcp)

# Explicitly register the tools
# register_financial_tools(tool_registry)

# import app.tools.clientview_financials as clientview_financials


# Confirm registered tools
tools = tool_registry.list_tools()
# logger.info(f"✅ Total tools registered: {len(tools)}")
print(f"✅ Total tools registered: {len(tools)}")

for name in tools:
    logger.info(f"🔧 Tool registered: {name}")

# Run the indexing
from app.config import config
from cohere_compass.clients.compass import CompassClient
from app.utils.setup_compass_index import index_mcp_tools


async def run_index():
    logger.info("Connecting to Compass...")
    client = CompassClient(
        index_url=config.COHERE_SERVER_URL,
        bearer_token=config.COHERE_SERVER_BEARER_TOKEN
    )

    logger.info("Running index_mcp_tools...")
    # await index_mcp_tools(client, config.COHERE_INDEX_NAME)
    await index_mcp_tools(client, config.COHERE_INDEX_NAME)

    logger.info("Indexing complete.")


# def test_index_mcp_tools():
#     asyncio.run(run_index())


# if __name__ == "__main__":
#     test_index_mcp_tools()


if __name__ == "__main__":
    asyncio.run(run_index()) 
