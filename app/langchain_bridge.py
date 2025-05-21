"""Experimental LangChain bridge.

This module mirrors ``MCPBridge`` but delegates planning to LangChain
agents. It keeps the same public interface so it can be swapped in place
of ``MCPBridge`` for experiments.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .mcp_bridge import MCPBridge
from .config import config

logger = logging.getLogger(__name__)


class LangChainBridge(MCPBridge):
    """Bridge implementation using LangChain planning."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        super().__init__()
        try:
            from langchain.agents import AgentType, Tool, initialize_agent
            from langchain.chat_models import ChatOpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "LangChain packages are required for LangChainBridge"
            ) from exc

        self._Tool = Tool
        self._initialize_agent = initialize_agent
        self._AgentType = AgentType
        self.llm = llm or ChatOpenAI(
            model_name=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=0,
        )

    async def _build_tools(self) -> List[Any]:
        """Return LangChain ``Tool`` objects wrapping registered MCP tools."""
        tools = []
        available = await self.get_available_tools()
        for name, tool in available.items():
            async def _fn(text: str, tool_name: str = name) -> Any:
                try:
                    params = json.loads(text) if text else {}
                except Exception:
                    params = {}
                return await self.execute_tool(tool_name, params)

            tools.append(
                self._Tool(
                    name=name,
                    func=_fn,
                    coroutine=_fn,
                    description=getattr(tool, "description", name),
                )
            )
        return tools

    async def plan_tool_chain(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate a plan using LangChain's Zero Shot agent."""
        tools = await self._build_tools()
        agent = self._initialize_agent(
            tools,
            self.llm,
            agent=self._AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )
        result = await agent.aplan(query)

        plan: List[Dict[str, Any]] = []
        for action, _ in result.get("intermediate_steps", []):
            name = getattr(action, "tool", None)
            if not name:
                continue
            args = (
                action.tool_input if isinstance(action.tool_input, dict) else {}
            )
            plan.append({"tool": name, "parameters": args})

        return self._validate_plan(plan)

    async def route_request(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route a request via LangChain planning with MCPBridge fallback."""
        try:
            plan = await self.plan_tool_chain(query, context)
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("LangChain planning failed: %s", exc)
            plan = []

        if not plan:
            return await super().route_request(query, context)

        step = plan[0]
        return {
            "intent": step["tool"],
            "endpoints": [
                {
                    "type": "tool",
                    "name": step["tool"],
                    "params": step.get("parameters", {}),
                }
            ],
            "confidence": 0.9,
            "prompt_type": "general",
        }
