import sys
from pathlib import Path
import types
import pytest

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub modules that may be missing in the test environment
sys.modules.setdefault('rbc_security', types.SimpleNamespace(enable_certs=lambda: None))
sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *a, **k: None))

class DummyAsyncClient:
    def __init__(self, *a, **kw):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def post(self, *a, **kw):
        class Resp:
            def __init__(self):
                self.status_code = 200
            def json(self):
                return {}
            def raise_for_status(self):
                pass
        return Resp()

sys.modules.setdefault('httpx', types.SimpleNamespace(AsyncClient=DummyAsyncClient))

# Provide a minimal mcp_server module so MCPBridge can import
class DummyMCP:
    async def get_tools(self):
        return {}

dummy_server = types.ModuleType('app.mcp_server')
dummy_server.mcp = DummyMCP()
sys.modules.setdefault('app.mcp_server', dummy_server)

import app.mcp_bridge as mcp_bridge
from app.mcp_bridge import MCPBridge, PlanStep

# Provide pydantic v2 compatibility when running on pydantic v1
if not hasattr(PlanStep, "model_validate"):
    PlanStep.model_validate = classmethod(lambda cls, data: cls.parse_obj(data))
if not hasattr(PlanStep, "model_dump"):
    PlanStep.model_dump = lambda self, *a, **k: {"tool": self.tool, "parameters": self.parameters}


@pytest.mark.asyncio
async def test_plan_success(monkeypatch):
    async def fake_call(messages):
        return '[{"tool": "search_docs", "parameters": {"query": "policy"}}]'
    monkeypatch.setattr(mcp_bridge, "_call_llm", fake_call)
    monkeypatch.setattr(MCPBridge, "_log_plan", lambda self, plan: None)
    bridge = MCPBridge()
    result = await bridge.plan_tool_chain("find policy")
    assert result == [{"tool": "search_docs", "parameters": {"query": "policy"}}]


@pytest.mark.asyncio
async def test_plan_fallback(monkeypatch):
    async def bad_call(messages):
        raise RuntimeError("fail")
    monkeypatch.setattr(mcp_bridge, "_call_llm", bad_call)
    monkeypatch.setattr(MCPBridge, "_log_plan", lambda self, plan: None)

    async def dummy_route(self, query, context=None):
        return {
            "intent": "server_info",
            "endpoints": [{"type": "tool", "name": "server_info", "params": {}}],
        }

    monkeypatch.setattr(MCPBridge, "route_request", dummy_route)
    bridge = MCPBridge()
    result = await bridge.plan_tool_chain("hi")
    assert result == [{"tool": "server_info", "parameters": {}}]
