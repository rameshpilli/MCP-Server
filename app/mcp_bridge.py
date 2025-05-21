"""
MCP Bridge: The Intelligence Layer Behind Tool Selection and Execution

The MCP Bridge acts as the "brain" of our system, intelligently connecting user requests to the
right tools. Think of it as an experienced concierge who knows exactly which service can fulfill
your needs.

How it works:
    1. When a user asks a question or makes a request, the Bridge analyzes what they're trying to do
    2. It looks through all available tools to find the best match for that intention
    3. It extracts relevant parameters from the user's request (like currency type, client names, etc.)
    4. It formulates the proper tool call with those parameters
    5. Finally, it returns the results in a user-friendly format

Behind the scenes, it uses:
    - Cohere Compass for semantic understanding of user intent
    - Parameter extraction using both LLM and rule-based approaches
    - Flexible matching to handle variations in terminology
    - Caching to improve performance for similar queries

This hybrid approach ensures robustness - the LLM provides flexibility in understanding user
requests, while the rule-based system offers reliability when the LLM might struggle.

The Bridge is designed to be extensible - new tools can be added without changing the core
routing logic, and parameter mappings can be updated without code changes.
"""
# app/mcp_bridge.py
import logging
from typing import Dict, Any, List, Optional
import time
import hashlib
import datetime
import json
from rbc_security import enable_certs
from .config import config
from .mcp_server_old import mcp
from app.utils.parameter_extractor import extract_parameters_with_llm, _call_llm
from .results_coordinator import ResultsCoordinator
from .fallback_handler import FallbackHandler
from .schemas import PlanStep

logger = logging.getLogger("mcp_server.bridge")
enable_certs()


class MCPBridge:
    def __init__(self):
        self.compass_api_url = getattr(config, 'COMPASS_API_URL', None)
        self.compass_bearer_token = getattr(config, 'COMPASS_BEARER_TOKEN', None)
        self.compass_index = getattr(config, 'COMPASS_INDEX_NAME', None)
        self.mcp = mcp
        self.compass_client = None

        self.results_coordinator = ResultsCoordinator(
            sqlite_cache=None if not config.USE_SQLITE_CACHE else None,
            vector_cache=None if not config.USE_VECTOR_DB else None,
        )
        self.fallback_handler = FallbackHandler(
            sqlite_cache=self.results_coordinator.sqlite_cache,
            vector_db=self.results_coordinator.vector_cache,
            use_sqlite=config.USE_SQLITE_CACHE,
            use_vector=config.USE_VECTOR_DB,
        )

        # Simple parameter extraction cache
        self._param_cache = {}
        self._cache_ttl = 300  # 5 minutes

        if self.compass_api_url and self.compass_bearer_token:
            try:
                from cohere_compass.clients.compass import CompassClient
                self.compass_client = CompassClient(
                    index_url=self.compass_api_url,
                    bearer_token=self.compass_bearer_token
                )
                logger.info("Compass client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Compass client: {e}")
        else:
            logger.warning("Compass not configured properly")

    async def get_available_tools(self):
        return await self.mcp.get_tools()

    async def route_request(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route a request to appropriate tools based on the message content.
        Primarily uses Compass for tool discovery with minimal fallbacks.
        """
        logger.info(f"Routing query: '{query}'")
        available_tools = await self.get_available_tools()
        tool_names = list(available_tools.keys())
        query_lower = query.lower()

        # Try Compass-based routing first if configured
        if self.compass_client and self.compass_index:
            try:
                logger.info("Using Compass for tool discovery")
                from asyncio import to_thread

                # Search Compass for relevant tools
                result = await to_thread(
                    lambda: self.compass_client.search_chunks(
                        index_name=self.compass_index,
                        query=query,
                        top_k=3  # Get top 3 matches
                    )
                )

                hits = getattr(result, 'hits', [])
                if hits:
                    # Process hits to find matching tools
                    for hit in hits:
                        content = getattr(hit, 'content', {})
                        tool_name = None

                        # Extract tool name from hit content (handle different content formats)
                        if isinstance(content, dict):
                            tool_name = content.get('name')
                            namespace = content.get('namespace')

                            # Construct full tool name if necessary
                            if tool_name and namespace:
                                full_tool_name = f"{namespace}:{tool_name}"
                            else:
                                full_tool_name = tool_name

                        # Find the actual registered tool that matches
                        matching_tool = None
                        if full_tool_name in tool_names:
                            matching_tool = full_tool_name
                        else:
                            # Try to match by name without namespace
                            matching_tools = [t for t in tool_names if tool_name in t]
                            if matching_tools:
                                matching_tool = matching_tools[0]

                        if matching_tool:
                            logger.info(f"Compass selected tool: {matching_tool}")

                            # Extract parameters for this tool
                            params = await self._get_tool_params(matching_tool, query)

                            return {
                                "intent": tool_name,
                                "endpoints": [{
                                    "type": "tool",
                                    "name": matching_tool,
                                    "params": params
                                }],
                                "confidence": max(0.7, min(0.95, float(getattr(hit, 'score', 0.7)))),
                                "prompt_type": self._determine_prompt_type(query)
                            }

                    logger.info("Compass found hits but no matching tools in registry")

            except Exception as e:
                logger.error(f"Compass routing failed: {e}")
                # Continue to fallbacks

        # Simple fallback for financial tools
        if any(keyword in query_lower for keyword in ["client", "top", "revenue", "financial", "money"]):
            # Look for different types of financial tools based on query content
            if "product" in query_lower:
                # Product breakdown request
                product_tools = [t for t in tool_names if any(name in t.lower() for name in
                                                              ["client_value_by_product", "get_client_value_by_product"])]
                if product_tools:
                    selected_tool = product_tools[0]
                    logger.info(f"Selected product tool: {selected_tool}")
                    return {
                        "intent": "financial_product",
                        "endpoints": [{
                            "type": "tool",
                            "name": selected_tool,
                            "params": await self._get_tool_params(selected_tool, query)
                        }],
                        "confidence": 0.75,
                        "prompt_type": "financial"
                    }

            if any(word in query_lower for word in ["time", "period", "year", "month", "quarter"]):
                # Time-based analysis request
                time_tools = [t for t in tool_names if any(name in t.lower() for name in
                                                           ["client_value_by_time", "get_client_value_by_time"])]
                if time_tools:
                    selected_tool = time_tools[0]
                    logger.info(f"Selected time tool: {selected_tool}")
                    return {
                        "intent": "financial_time",
                        "endpoints": [{
                            "type": "tool",
                            "name": selected_tool,
                            "params": await self._get_tool_params(selected_tool, query)
                        }],
                        "confidence": 0.75,
                        "prompt_type": "financial"
                    }

            # Default to top clients tool for other financial queries
            top_client_tools = [t for t in tool_names if any(name in t.lower() for name in
                                                             ["top_clients", "get_top_clients"])]
            if top_client_tools:
                selected_tool = top_client_tools[0]
                logger.info(f"Selected top clients tool: {selected_tool}")
                return {
                    "intent": "financial",
                    "endpoints": [{
                        "type": "tool",
                        "name": selected_tool,
                        "params": await self._get_tool_params(selected_tool, query)
                    }],
                    "confidence": 0.75,
                    "prompt_type": "financial"
                }

        # Final fallback - use server_info or list_tools or health_check
        logger.info("No tool matches found, using fallback")
        for fallback_name in ["server_info", "health_check", "list_tools"]:
            fallback_tools = [t for t in tool_names if fallback_name in t.lower()]
            if fallback_tools:
                return {
                    "intent": "fallback",
                    "endpoints": [{
                        "type": "tool",
                        "name": fallback_tools[0],
                        "params": {}
                    }],
                    "confidence": 0.3,
                    "prompt_type": "general"
                }

        # If no fallback tool found, use the first available tool
        return {
            "intent": "fallback",
            "endpoints": [{
                "type": "tool",
                "name": tool_names[0] if tool_names else "server_info",
                "params": {}
            }],
            "confidence": 0.3,
            "prompt_type": "general"
        }

    async def route(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route method to handle client calls expecting mcp.route

        This is a compatibility method that simply calls route_request
        """
        # Simply delegate to the existing route_request method
        return await self.route_request(message, context or {})

    # PARAMETERS TO BE EXTRACTED FROM USER PROMPT USING LLM
    def _get_tool_from_registry(self, tool_name: str):
        """Helper method to get tool from registry with namespace handling"""
        from app.registry import tool_registry

        try:
            # Handle tools with explicit namespace
            if ":" in tool_name:
                namespace, name = tool_name.split(":", 1)
                return tool_registry.get_tool(name, namespace=namespace)

            # Try common namespaces
            for namespace in ["clientview", "crm", "default"]:
                try:
                    tool = tool_registry.get_tool(tool_name, namespace=namespace)
                    if tool:
                        return tool
                except Exception:
                    pass

            # Try without namespace
            return tool_registry.get_tool(tool_name)
        except Exception as e:
            logger.warning(f"Error getting tool '{tool_name}': {e}")
            return None

    def _get_parameter_mappings(self):
        """
        Get mappings for parameter values.
        Centralizes parameter term definitions for easy updates.
        """
        try:
            from app.utils.parameter_mappings import load_parameter_mappings
            return load_parameter_mappings()
        except Exception as e:
            logger.warning(f"Failed to load external parameter mappings: {e}")
            from app.utils.parameter_mappings import DEFAULT_MAPPINGS
            return DEFAULT_MAPPINGS

    def _normalize_parameters(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters based on schema definitions and standard values.
        This is more flexible than hard-coded normalization.
        """
        normalized = {}

        # Get parameter mappings
        param_mappings = self._get_parameter_mappings()

        # Process each parameter
        for param_name, param_value in params.items():
            # Skip if value is None
            if param_value is None:
                continue

            # Check if parameter exists in schema
            if param_name not in schema:
                continue

            # Special handling for different parameter types
            if param_name in ["currency", "ccy_code"]:
                # Convert to string and uppercase
                value_str = str(param_value).upper()
                # Find the standard value
                for std_value, terms in param_mappings.get("currency", {}).items():
                    if value_str == std_value or value_str in [t.upper() for t in terms]:
                        normalized[param_name] = std_value
                        break
                else:
                    # Default to USD if no match
                    normalized[param_name] = "USD"

            elif param_name in ["sorting", "sorting_criteria"]:
                value_str = str(param_value).lower()
                for std_value, terms in param_mappings.get("sorting", {}).items():
                    if value_str == std_value or value_str in terms:
                        normalized[param_name] = std_value
                        break
                else:
                    # Default to "top" if no match
                    normalized[param_name] = "top"

            elif param_name == "region":
                value_str = str(param_value).upper()
                for std_value, terms in param_mappings.get("region", {}).items():
                    if value_str == std_value or value_str in [t.upper() for t in terms]:
                        normalized[param_name] = std_value
                        break

            elif param_name in ["time_period_year"]:
                # Ensure it's an integer and within a reasonable range
                try:
                    year = int(param_value)
                    # Allow 3 years back and 1 year forward
                    current_year = datetime.now().year
                    if current_year - 3 <= year <= current_year + 1:
                        normalized[param_name] = year
                    else:
                        normalized[param_name] = current_year  # Default to current year
                except (ValueError, TypeError):
                    normalized[param_name] = datetime.now().year  # Default to current year

            else:
                # For other parameters, just use the value as is
                normalized[param_name] = param_value

        return normalized

    def _extract_parameter_value(self, param_name, query_lower, param_mappings):
        """
        Extract value for a specific parameter based on mappings.
        Handles different parameter names that might be used for the same concept.
        """
        # Map common parameter name variations to standard names
        param_name_mapping = {
            "currency": ["currency", "ccy_code", "ccy"],
            "sorting": ["sorting", "sorting_criteria", "sort", "order_by"],
            "region": ["region", "region_filter"],
            "time_period": ["time_period"],
            "time_filter": ["time_filter"],
            "focus_list": ["focus_list", "focus_list_filter"],
            "client_cdrid": ["client_cdrid", "client_id", "cdrid"]
        }

        # Find the normalized parameter name
        normalized_name = None
        for std_name, variants in param_name_mapping.items():
            if param_name in variants:
                normalized_name = std_name
                break

        # If we don't have mappings for this parameter, return None
        if not normalized_name or normalized_name not in param_mappings:
            return None

        # Check for matches in the query
        mappings = param_mappings.get(normalized_name, {})
        for value, terms in mappings.items():
            if any(term in query_lower for term in terms):
                return value

        # Special handling for numeric parameters
        if normalized_name == "client_cdrid":
            import re
            # Look for numbers that might be client IDs
            match = re.search(r'\b\d{5,}\b', query_lower)
            if match:
                return int(match.group())

        # Return None if no match found
        return None

    async def _get_tool_params(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Extract parameters from the user's query for a specific tool.
        Uses LLM with fallback to rule-based extraction.
        Designed to be flexible and extensible.
        """
        from app.registry import tool_registry
        import re
        query_lower = query.lower()

        # Log the start of parameter extraction
        logger.info(f"Starting parameter extraction for tool '{tool_name}'")
        logger.info(f"Query: '{query}'")

        # Step 1: Check cache
        cache_key = self._get_cache_key(tool_name, query)
        cached_params = self._check_param_cache(cache_key)

        if cached_params:
            logger.info(f"Using cached parameters: {cached_params}")
            return cached_params

        # Step 2: Get tool schema
        tool = self._get_tool_from_registry(tool_name)
        if not tool:
            logger.warning(f"Tool '{tool_name}' not found in registry")
            return {}

        # Get the input schema
        input_schema = getattr(tool, 'input_schema', {})
        logger.info(f"Tool input schema has {len(input_schema)} parameters")

        # Step 3: Try LLM-based extraction first
        try:
            llm_params = await extract_parameters_with_llm(query, tool_name, input_schema)
            if llm_params:
                logger.info(f"LLM extracted parameters: {llm_params}")

                # Normalize the parameters based on schema
                normalized_params = self._normalize_parameters(llm_params, input_schema)
                logger.info(f"Normalized parameters: {normalized_params}")

                # Save to cache and return
                self._update_param_cache(cache_key, normalized_params)
                return normalized_params
        except Exception as e:
            logger.warning(f"LLM parameter extraction failed: {e}")

        # Step 4: Fall back to rule-based extraction
        logger.info("Falling back to rule-based parameter extraction")

        # Get parameter term mappings from a centralized configuration
        param_mappings = self._get_parameter_mappings()

        extracted_params = {}

        # Process each parameter in the schema
        for param_name in input_schema.keys():
            # Special handling based on parameter type
            param_value = self._extract_parameter_value(param_name, query_lower, param_mappings)
            if param_value is not None:
                extracted_params[param_name] = param_value

        # Normalize and validate extracted parameters
        normalized_params = self._normalize_parameters(extracted_params, input_schema)

        # Save to cache if we have any parameters
        if normalized_params:
            self._update_param_cache(cache_key, normalized_params)
            logger.info(f"Rule-based extracted parameters: {normalized_params}")
        else:
            logger.info("No parameters extracted")

        return normalized_params

    def _get_cache_key(self, tool_name: str, query: str) -> str:
        """Generate a cache key for parameter extraction"""
        # Create a unique key from the tool name and query
        combined = f"{tool_name}:{query.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _check_param_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if parameters are cached and not expired"""
        if cache_key in self._param_cache:
            timestamp, params = self._param_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return params
        return None

    def _update_param_cache(self, cache_key: str, params: Dict[str, Any]) -> None:
        """Update the parameter cache"""
        self._param_cache[cache_key] = (time.time(), params)

        # Clean up old entries if the cache gets too large
        if len(self._param_cache) > 100:  # Limit cache size
            expired_keys = []
            current_time = time.time()
            for key, (timestamp, _) in self._param_cache.items():
                if current_time - timestamp > self._cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._param_cache[key]

    def _determine_prompt_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(w in query_lower for w in ["explain", "how", "what", "why"]):
            return "explanation"
        if "summarize" in query_lower:
            return "summarization"
        if "analyze" in query_lower or "sentiment" in query_lower:
            return "analysis"
        if "pdf" in query_lower or "transcript" in query_lower:
            return "document"
        return "general"

    def _extract_doc_name(self, query: str) -> str:
        for word in query.split():
            if word.endswith(('.md', '.txt', '.pdf')):
                return word
        return "README.md"

    async def _filter_valid_tools(self, hits, tool_names: List[str], query: str, max_tools: int = 1) -> List[
        Dict[str, Any]]:
        """
        Filters Compass hits and returns a list of valid tools with parameters.
        Args:
            hits: Compass search results
            tool_names: List of registered tool names
            query: Original user query
            max_tools: Max tools to return (default: 1)

        Returns:
            List of valid tool endpoint dicts
        """
        top_tools = []
        seen = set()

        for hit in hits:
            content = getattr(hit, 'content', {})
            tool_name = content.get('name') if isinstance(content, dict) else None

            if tool_name and tool_name not in seen:
                # Check if the tool exists in any namespace
                matching_tools = [t for t in tool_names if tool_name in t]

                if matching_tools:
                    actual_tool = matching_tools[0]
                    params = await self._get_tool_params(actual_tool, query)
                    if params:
                        top_tools.append({
                            "type": "tool",
                            "name": actual_tool,
                            "params": params
                        })
                        seen.add(tool_name)
                        seen.add(actual_tool)

            if len(top_tools) >= max_tools:
                break

        return top_tools

    async def plan_tool_chain(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate a tool execution plan using the LLM."""
        planning_prompt = (
            f"Given this query: \"{query}\"\n"
            "Return a JSON array of tool calls needed to satisfy it."
        )

        try:
            response = await _call_llm([
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": planning_prompt},
            ])
            plan = json.loads(response)
            if isinstance(plan, list):
                plan = self._validate_plan(plan)
                if plan:
                    self._log_plan(plan)
                    return plan
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
        # Fallback to single step using legacy routing
        route = await self.route_request(query, context or {})
        step = {"tool": route["endpoints"][0]["name"], "parameters": route["endpoints"][0].get("params", {})}
        plan = [step]
        self._log_plan(plan)
        return plan

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]):
        """Execute a tool registered with the MCP server."""
        tools = await self.mcp.get_tools()
        tool = tools.get(tool_name)
        if not tool:
            return None
        ctx = type("Tmp", (), {"request_context": type("TmpCtx", (), {"context": {}})()})()
        if params:
            return await tool.fn(ctx, **params)
        return await tool.fn(ctx)

    async def run_plan(self, plan: List[Dict[str, Any]]):
        """Execute a previously generated plan."""
        return await self.results_coordinator.process_plan(plan, self)

    async def run_plan_streaming(self, plan: List[Dict[str, Any]]):
        """Yield step results one by one for streaming."""
        async for res in self.results_coordinator.process_plan_streaming(plan, self):
            yield res

    def _validate_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate plan steps using the PlanStep schema."""
        valid_steps = []
        for step in plan:
            try:
                obj = PlanStep.model_validate(step)
                valid_steps.append(obj.model_dump())
            except Exception as exc:
                logger.warning(f"Invalid plan step dropped: {exc}")
        return valid_steps

    def _log_plan(self, plan: List[Dict[str, Any]]):
        """Write plan JSON to disk for debugging."""
        try:
            output_path = config.PLAN_LOG_FILE
            with open(output_path, "w") as f:
                json.dump(plan, f, indent=2)
            logger.debug(f"Saved plan to {output_path}")
        except Exception as exc:
            logger.warning(f"Failed to log plan: {exc}")
