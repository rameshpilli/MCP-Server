import asyncio
from unittest.mock import patch
from app.mcp_bridge import MCPBridge

async def show_success():
    async def fake_call(messages):
        return '[{"tool": "search_docs", "parameters": {"query": "policy"}}]'

    with patch.object(MCPBridge, "_call_llm", side_effect=fake_call), \
         patch.object(MCPBridge, "_log_plan", lambda self, plan: None):

        bridge = MCPBridge()
        result = await bridge.plan_tool_chain("find policy")
        print("Success:", result)

async def show_fallback():
    async def bad_call(messages):
        raise RuntimeError("fail")

    async def dummy_route(self, query, context=None):
        return {
            "intent": "server_info",
            "endpoints": [{"type": "tool", "name": "server_info", "params": {}}],
        }

    with patch.object(MCPBridge, "_call_llm", side_effect=bad_call), \
         patch.object(MCPBridge, "_log_plan", lambda self, plan: None), \
         patch.object(MCPBridge, "route_request", dummy_route):

        bridge = MCPBridge()
        result = await bridge.plan_tool_chain("hi")
        print("Fallback:", result)

asyncio.run(show_success())
asyncio.run(show_fallback())
