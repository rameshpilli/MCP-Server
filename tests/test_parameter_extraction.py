import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent))
from app.mcp_bridge import MCPBridge


async def run_test():
    bridge = MCPBridge()
    params = await bridge.extract_parameters(
        "Show top clients in CAD",
        "crm:get_top_clients",
    )
    assert params.get("currency") == "CAD"


def test_extract_parameters():
    asyncio.run(run_test())
