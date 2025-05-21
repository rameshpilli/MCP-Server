"""Quick comparison of planning between MCPBridge and LangChainBridge."""

import asyncio
from app.mcp_bridge import MCPBridge
from app.langchain_bridge import LangChainBridge


async def main() -> None:
    bridge = MCPBridge()
    lc_bridge = LangChainBridge()

    queries = [
        "Top clients in Canada",
        "Show me the README document",
    ]

    for q in queries:
        plan_a = await bridge.plan_tool_chain(q)
        plan_b = await lc_bridge.plan_tool_chain(q)
        print("Query:", q)
        print("MCPBridge plan:", plan_a)
        print("LangChain plan:", plan_b)
        print()


if __name__ == "__main__":
    asyncio.run(main())
