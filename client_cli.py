import argparse
import asyncio
from mcp_client import MCPClient


def main():
    parser = argparse.ArgumentParser(description="Query the MCP server")
    parser.add_argument("--query", required=True, help="Query to send")
    parser.add_argument("--base-url", default=None, help="MCP server base URL")
    args = parser.parse_args()

    client = MCPClient(base_url=args.base_url)
    result = asyncio.run(client.query(args.query))
    print(result)


if __name__ == "__main__":
    main()

