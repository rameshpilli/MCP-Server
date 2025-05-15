import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from app.mcp_server import mcp

async def check_fastmcp_tools():
    tools = await mcp.get_tools()
    print("Tools registered with FastMCP:")
    for name, tool in tools.items():
        print(f"  - {name}")
        print(f"    Function: {tool.fn}")
        print(f"    Description: {getattr(tool, 'description', 'No description')}")
        print()

if __name__ == "__main__":
    asyncio.run(check_fastmcp_tools()) 