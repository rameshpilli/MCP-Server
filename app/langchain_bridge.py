"""
LangChainBridge:
    - Experimental bridge that uses LangChain agents to plan tool execution.
    - Unlike MCPBridge, this uses our internal LLM (via LLMWrapper) instead of OpenAI.
    - Fallbacks to traditional Compass-based routing if agent planning fails.

Why we use it:
    - LangChain lets us intelligently route and chain tools based on user intent.
    - We override their LLM usage with our internal bearer-token-based endpoint.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .llm_wrapper import LLMWrapper
from .mcp_bridge import MCPBridge
from app.client import mcp_client
from .config import config

logger = logging.getLogger(__name__)


class LangChainBridge(MCPBridge):
    """Bridge implementation using LangChain planning."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        super().__init__()
        try:
            from langchain.agents import AgentType, Tool, initialize_agent
            from langchain_community.chat_models import ChatOpenAI
        except Exception as exc:
            raise ImportError("LangChain packages are required for LangChainBridge") from exc

        # LangChain agent configuration
        self._Tool = Tool
        self._initialize_agent = initialize_agent
        self._AgentType = AgentType

        # Inject our internal LLM (via wrapper) instead of OpenAI
        self.llm = llm or LLMWrapper(call_llm_fn=mcp_client.call_llm)

    async def _build_tools(self) -> List[Any]:
        """Wrap all MCP tools as LangChain-compatible tools."""
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
        """Use LangChain to generate a list of tools to run for this query."""
        tools = await self._build_tools()
        agent = self._initialize_agent(
            tools,
            self.llm,
            agent=self._AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )

        logger.info(f"Planning tool chain using LangChain for query: '{query}'")
        try:
            result = await agent.aplan(query)
            logger.info(f"LangChain agent returned: {result}")
        except Exception as e:
            logger.error(f"LangChain agent failed: {e}")
            raise

        # Parse tool calls from agent response
        plan: List[Dict[str, Any]] = []
        for action, _ in result.get("intermediate_steps", []):
            name = getattr(action, "tool", None)
            if not name:
                continue
            args = action.tool_input if isinstance(action.tool_input, dict) else {}
            plan.append({"tool": name, "parameters": args})

        return self._validate_plan(plan)

    async def route_request(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Try LangChain planning first. If it fails or returns nothing, fall back to normal Compass-based routing.
        """
        try:
            plan = await self.plan_tool_chain(query, context)
        except Exception as exc:
            logger.warning("LangChain planning failed: %s", exc)
            plan = []

        if not plan:
            logger.info("LangChain returned no plan — falling back to Compass routing.")
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
