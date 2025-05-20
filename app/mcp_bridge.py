"""
MCP Bridge:
    - Connects to Cohere Compass for intent recognition
    - Routes queries to appropriate tools
    - Extracts parameters from queries using an LLM
"""

import logging
import time
import hashlib
from typing import Dict, Any, List, Optional

from rbc_security import enable_certs

from .config import config
from .mcp_server import mcp
from app.utils.parameter_extractor import extract_parameters_with_llm

logger = logging.getLogger("mcp_server.bridge")
enable_certs()


class MCPBridge:
    def __init__(self):
        self.compass_api_url = getattr(config, "COMPASS_API_URL", None)
        self.compass_bearer_token = getattr(config, "COMPASS_BEARER_TOKEN", None)
        self.compass_index = getattr(config, "COMPASS_INDEX_NAME", None)
        self.mcp = mcp
        self.compass_client = None

        # Cache for parameter extraction
        self._param_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl = 300  # seconds


        if self.compass_api_url and self.compass_bearer_token:
            try:
                from cohere_compass.clients.compass import CompassClient

                self.compass_client = CompassClient(
                    index_url=self.compass_api_url,
                    bearer_token=self.compass_bearer_token,
                )
                logger.info("Compass client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Compass client: {e}")
        else:
            logger.warning("Compass not configured properly")

    async def get_available_tools(self):
        return await self.mcp.get_tools()

    async def route_request(self, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Route a request to appropriate tools."""
        logger.info(f"Routing query: '{query}'")
        available_tools = await self.get_available_tools()
        tool_names = list(available_tools.keys())
        query_lower = query.lower()

        # Use Compass if available
        if self.compass_client and self.compass_index:
            try:
                from asyncio import to_thread

                result = await to_thread(
                    lambda: self.compass_client.search_chunks(
                        index_name=self.compass_index,
                        query=query,
                        top_k=3,
                    )
                )
                hits = getattr(result, "hits", [])
                for hit in hits:
                    content = getattr(hit, "content", {})
                    tool_name = None
                    full_tool_name = None
                    if isinstance(content, dict):
                        tool_name = content.get("name")
                        ns = content.get("namespace")
                        if tool_name and ns:
                            full_tool_name = f"{ns}:{tool_name}"
                        else:
                            full_tool_name = tool_name
                    if not tool_name:
                        continue

                    matching_tool = None
                    if full_tool_name and full_tool_name in tool_names:
                        matching_tool = full_tool_name
                    else:
                        mt = [t for t in tool_names if tool_name in t]
                        if mt:
                            matching_tool = mt[0]
                    if matching_tool:
                        params = await self._get_tool_params(matching_tool, query)
                        return {
                            "intent": tool_name,
                            "endpoints": [
                                {"type": "tool", "name": matching_tool, "params": params}
                            ],
                            "confidence": max(0.7, min(0.95, float(getattr(hit, "score", 0.7)))),
                            "prompt_type": self._determine_prompt_type(query),
                        }
                logger.info("Compass found hits but no matching tools")
            except Exception as e:
                logger.error(f"Compass routing failed: {e}")

        # Simple fallback for financial queries
        if any(k in query_lower for k in ["client", "revenue", "financial", "top"]):
            top_tools = [t for t in tool_names if "get_top_clients" in t or "top_clients" in t]
            if top_tools:
                selected = top_tools[0]
                params = await self._get_tool_params(selected, query)
                return {
                    "intent": "financial",
                    "endpoints": [
                        {"type": "tool", "name": selected, "params": params}
                    ],
                    "confidence": 0.75,
                    "prompt_type": "financial",
                }

        # Final fallback
        for name in ["server_info", "health_check", "list_tools"]:
            fl = [t for t in tool_names if name in t.lower()]
            if fl:
                return {
                    "intent": "fallback",
                    "endpoints": [{"type": "tool", "name": fl[0], "params": {}}],
                    "confidence": 0.3,
                    "prompt_type": "general",
                }

        return {
            "intent": "fallback",
            "endpoints": [
                {"type": "tool", "name": tool_names[0] if tool_names else "server_info", "params": {}}
            ],
            "confidence": 0.3,
            "prompt_type": "general",
        }

    async def route(self, message: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return await self.route_request(message, context or {})

    def _get_tool_from_registry(self, tool_name: str):
        from app.registry import tool_registry

        try:
            if ":" in tool_name:
                ns, name = tool_name.split(":", 1)
                return tool_registry.get_tool(name, namespace=ns)
            for ns in ["clientview", "crm", "default", "finance"]:
                try:
                    tool = tool_registry.get_tool(tool_name, namespace=ns)
                    if tool:
                        return tool
                except Exception:
                    pass
            return tool_registry.get_tool(tool_name)
        except Exception as e:
            logger.warning(f"Error getting tool '{tool_name}': {e}")
            return None

    async def _get_tool_params(self, tool_name: str, query: str) -> Dict[str, Any]:
        from app.registry import tool_registry

        cache_key = self._get_cache_key(tool_name, query)
        cached = self._check_param_cache(cache_key)
        if cached:
            return cached

        tool = self._get_tool_from_registry(tool_name)
        if not tool:
            return {}
        schema = getattr(tool, "input_schema", {})
        if not schema:
            return {}

        params = await extract_parameters_with_llm(query, tool_name, schema)

        if params:
            self._update_param_cache(cache_key, params)
        return params

    def _get_cache_key(self, tool_name: str, query: str) -> str:
        combined = f"{tool_name}:{query.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _check_param_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._param_cache:
            ts, params = self._param_cache[key]
            if time.time() - ts < self._cache_ttl:
                return params
        return None

    def _update_param_cache(self, key: str, params: Dict[str, Any]) -> None:
        self._param_cache[key] = (time.time(), params)
        if len(self._param_cache) > 100:
            current = time.time()
            to_delete = [k for k, (ts, _) in self._param_cache.items() if current - ts > self._cache_ttl]
            for k in to_delete:
                self._param_cache.pop(k, None)

    def _determine_prompt_type(self, query: str) -> str:
        ql = query.lower()
        if any(w in ql for w in ["explain", "how", "what", "why"]):
            return "explanation"
        if "summarize" in ql:
            return "summarization"
        if "analyze" in ql or "sentiment" in ql:
            return "analysis"
        if "pdf" in ql or "transcript" in ql:
            return "document"
        return "general"
