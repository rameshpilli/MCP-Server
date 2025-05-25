import types
import sys
import asyncio
from importlib import import_module
from pathlib import Path

import pytest


def setup_module():
    """Create stub dependencies so app.mcp_server imports."""
    sys.path.append(str(Path(__file__).parent.parent))
    # Stub fastmcp
    fastmcp = types.ModuleType('fastmcp')

    class FakeFastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        async def get_tools(self):
            return {}

    class Context:
        pass

    fastmcp.FastMCP = FakeFastMCP
    fastmcp.Context = Context
    sys.modules['fastmcp'] = fastmcp

    # Stub sse_starlette.sse.EventSourceResponse
    sse = types.ModuleType('sse_starlette.sse')
    class DummyResponse:
        pass
    sse.EventSourceResponse = DummyResponse
    sys.modules['sse_starlette.sse'] = sse
    sys.modules['sse_starlette'] = types.ModuleType('sse_starlette')

    # Stub dotenv.load_dotenv
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = dotenv


def test_server_info_grouping(monkeypatch):
    module = import_module('app.mcp_server')

    class Tool:
        def __init__(self, description):
            self.description = description

    tools = {
        'alpha:one': Tool('first'),
        'alpha:two': Tool('second'),
        'beta:main': Tool('beta tool'),
    }

    async def fake_get_tools():
        return tools

    monkeypatch.setattr(module.mcp, 'get_tools', fake_get_tools)

    output = asyncio.run(module.server_info(None))

    assert '## alpha' in output
    assert '## beta' in output
    alpha_section = output.split('## alpha')[1].split('## beta')[0]
    assert '`alpha:one`' in alpha_section
    assert '`alpha:two`' in alpha_section
    beta_section = output.split('## beta')[1]
    assert '`beta:main`' in beta_section
