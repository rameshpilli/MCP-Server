import asyncio
import os 
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from app.mcp_server import mcp

async def list_tools():
    tools = await mcp.get_tools()
    print(f"Found {len(tools)} tools:")
    for name in sorted(tools.keys()):
        print(f"  - {name}")
    
    print("\nLooking for clientview tools:")
    clientview_tools = [name for name in tools.keys() if 'client' in name.lower()]
    if clientview_tools:
        print(f"Found {len(clientview_tools)} clientview tools:")
        for name in clientview_tools:
            print(f"  - {name}")
    else:
        print("No clientview tools found.")

if __name__ == "__main__":
    asyncio.run(list_tools()) 