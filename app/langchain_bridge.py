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
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
import inspect

from .llm_wrapper import LLMWrapper
from .mcp_bridge import MCPBridge
from .config import config
from .utils.parameter_extractor import extract_parameters_with_llm

logger = logging.getLogger(__name__)


class LangChainBridge(MCPBridge):
    """Bridge implementation using LangChain planning."""

    def __init__(self, mcp=None, llm: Optional[Any] = None) -> None:
        super().__init__(mcp)
        try:
            from langchain.agents import AgentType, Tool, initialize_agent
            from langchain.agents.agent import AgentExecutor
            from langchain.agents.format_scratchpad import format_to_openai_function_messages
            from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain.tools.render import format_tool_to_openai_function
        except Exception as exc:
            raise ImportError("LangChain packages are required for LangChainBridge") from exc

        # Store LangChain imports for later use
        self._Tool = Tool
        self._initialize_agent = initialize_agent
        self._AgentType = AgentType
        self._AgentExecutor = AgentExecutor
        self._format_to_openai_function_messages = format_to_openai_function_messages
        self._OpenAIFunctionsAgentOutputParser = OpenAIFunctionsAgentOutputParser
        self._ChatPromptTemplate = ChatPromptTemplate
        self._MessagesPlaceholder = MessagesPlaceholder
        self._format_tool_to_openai_function = format_tool_to_openai_function

        # Inject our internal LLM (via wrapper) instead of OpenAI
        if llm is None:
            from app.client import mcp_client  # local import to avoid cycle
            llm = LLMWrapper(call_llm_fn=mcp_client.call_llm)
        self.llm = llm
        
        # Cache for tool schemas
        self._tool_schemas_cache = {}

    async def _get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a specific tool."""
        if tool_name in self._tool_schemas_cache:
            return self._tool_schemas_cache[tool_name]
        
        # Get tools directly from FastMCP
        if self.mcp:
            mcp_tools = await self.mcp.get_tools()
            
            # Look for the tool by name (handle namespaced names)
            actual_tool_name = tool_name
            if ":" in tool_name:
                actual_tool_name = tool_name.split(":", 1)[1]
            
            if actual_tool_name in mcp_tools:
                tool_obj = mcp_tools[actual_tool_name]
                schema = {
                    "type": "object",
                    "properties": {}
                }
                
                # Extract schema from function signature (safer approach)
                if hasattr(tool_obj, 'fn'):
                    try:
                        sig = inspect.signature(tool_obj.fn)
                        
                        # Skip first parameter (usually ctx/context)
                        params = list(sig.parameters.items())
                        if params and params[0][0] in ('ctx', 'context'):
                            params = params[1:]
                        
                        for param_name, param in params:
                            param_info = {
                                "type": "string"  # Default to string
                            }
                            
                            # Get annotation if available
                            if param.annotation != inspect.Parameter.empty:
                                if param.annotation == str:
                                    param_info["type"] = "string"
                                elif param.annotation == int:
                                    param_info["type"] = "integer"
                                elif param.annotation == float:
                                    param_info["type"] = "number"
                                elif param.annotation == bool:
                                    param_info["type"] = "boolean"
                                elif hasattr(param.annotation, "__name__"):
                                    # Handle other types
                                    param_info["type"] = "string"
                            
                            # Get default if available
                            if param.default != inspect.Parameter.empty:
                                param_info["default"] = param.default
                            
                            schema["properties"][param_name] = param_info
                    except Exception as e:
                        logger.debug(f"Error extracting schema from function signature: {e}")
                
                self._tool_schemas_cache[tool_name] = schema
                return schema
        
        # Default empty schema if not found
        default_schema = {"type": "object", "properties": {}}
        self._tool_schemas_cache[tool_name] = default_schema
        return default_schema

    async def _build_tools(self) -> List[Any]:
        """Wrap all MCP tools as LangChain-compatible tools."""
        tools = []
        
        try:
            # Get tools directly from FastMCP - this returns {tool_name: Tool} dict
            available_tools = await self.mcp.get_tools()
            logger.info(f"Found {len(available_tools)} tools from FastMCP")

            for tool_name, tool_obj in available_tools.items():
                logger.debug(f"Processing tool: {tool_name}")
                # Get description from the tool object
                description = getattr(tool_obj, 'description', f"Tool: {tool_name}")
                
                # Create a closure to capture the tool name
                async def _create_tool_fn(tool_name=tool_name):
                    async def _fn(text: str) -> Any:
                        """Execute the tool with parameters extracted from text."""
                        try:
                            # Try to parse as JSON first
                            try:
                                params = json.loads(text) if text else {}
                            except Exception:
                                # If not JSON, try to extract parameters using LLM
                                schema = await self._get_tool_schema(tool_name)
                                params = await extract_parameters_with_llm(text, schema, tool_name)
                            
                            # Execute the tool
                            result = await self.execute_tool(tool_name, params)
                            return result
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            return f"Error executing tool {tool_name}: {str(e)}"
                    
                    return _fn
                
                # Create the tool function
                tool_fn = await _create_tool_fn()
                
                # Get tool schema for better function descriptions
                # schema = await self._get_tool_schema(tool_name)
                # logger.debug(f"Tool {tool_name} schema: {schema}")
                
                # Create LangChain tool without args_schema for now to avoid bound method error
                lc_tool = self._Tool(
                    name=tool_name,
                    func=tool_fn,
                    coroutine=tool_fn,
                    description=description
                    # args_schema=schema  # Removing this to avoid the bound method error
                )
                
                tools.append(lc_tool)
                logger.debug(f"Created LangChain tool for: {tool_name}")

            logger.info(f"Successfully built {len(tools)} LangChain tools")
            return tools
        except Exception as e:
            logger.error(f"Error in _build_tools: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    async def plan_tool_chain(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Use LangChain to generate a list of tools to run for this query."""
        if context is None:
            context = {}
            
        try:
            tools = await self._build_tools()
            if not tools:
                logger.warning("No tools available for planning")
                return []
                
            # Create a better prompt template for the agent
            system_prompt = """You are an expert assistant that helps users by using tools when needed. 
            You have access to the following tools:
            
            {tools}
            
            When a user asks a question, analyze it carefully to determine if you need to use tools to answer it.
            If multiple tools are needed, use them in sequence to build a complete answer.
            Always try to use the most specific tool that matches the user's intent.
            If no tools are needed, just say so.
            """
            
            human_prompt = "{input}"
            
            # Create prompt template
            prompt = self._ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt),
                self._MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Convert tools to OpenAI functions
            llm_with_tools = self.llm.bind(
                functions=[self._format_tool_to_openai_function(t) for t in tools]
            )
            
            # Create the agent
            agent = (
                {
                    "input": lambda x: x["input"],
                    "tools": lambda x: [tool.name + ": " + tool.description for tool in tools],  # Add tools info
                    "agent_scratchpad": lambda x: self._format_to_openai_function_messages(x["intermediate_steps"])
                }
                | prompt
                | llm_with_tools
                | self._OpenAIFunctionsAgentOutputParser()
            )
            
            # Create the executor
            agent_executor = self._AgentExecutor(agent=agent, tools=tools, verbose=True)

            logger.info(f"Planning tool chain using LangChain for query: '{query}'")
            try:
                # Add context to the query if available
                input_with_context = query
                if context:
                    context_str = json.dumps(context, indent=2)
                    input_with_context = f"{query}\n\nContext: {context_str}"
                
                logger.debug(f"Input with context: {input_with_context}")
                logger.debug(f"Tools for prompt: {[f'{tool.name}: {tool.description}' for tool in tools]}")
                
                # Set debug logging for LangChain to see what's happening
                import logging
                langchain_logger = logging.getLogger('langchain')
                langchain_logger.setLevel(logging.DEBUG)
                
                # Also set our own logger to debug
                logger.setLevel(logging.DEBUG)
                # Set handler to console if not already set
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    handler.setLevel(logging.DEBUG)
                    logger.addHandler(handler)
                    
                # Execute the agent
                result = await agent_executor.ainvoke({
                    "input": input_with_context,
                    "tools": [f"{tool.name}: {tool.description}" for tool in tools]
                })
                logger.info(f"LangChain agent returned: {result}")
            except Exception as e:
                logger.error(f"LangChain agent failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return []

            # Parse tool calls from agent response
            plan: List[Dict[str, Any]] = []
            for action, action_result in result.get("intermediate_steps", []):
                tool_name = getattr(action, "tool", None)
                if not tool_name:
                    continue
                    
                # Get tool input (parameters)
                tool_input = action.tool_input
                if isinstance(tool_input, str):
                    try:
                        # Try to parse as JSON
                        params = json.loads(tool_input)
                    except Exception:
                        # If not JSON, use as is
                        params = {"text": tool_input}
                elif isinstance(tool_input, dict):
                    params = tool_input
                else:
                    params = {}
                    
                # Add to plan
                plan.append({
                    "tool": tool_name,
                    "parameters": params,
                    "result": action_result
                })

            return self._validate_plan(plan)
        except Exception as e:
            logger.error(f"Error in plan_tool_chain: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _validate_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate the plan to ensure it's executable."""
        if not plan:
            return []
            
        # Filter out any invalid tools
        valid_plan = []
        for step in plan:
            if "tool" not in step:
                logger.warning(f"Invalid plan step, missing 'tool': {step}")
                continue
                
            # Ensure parameters is a dict
            if "parameters" not in step or not isinstance(step["parameters"], dict):
                step["parameters"] = {}
                
            valid_plan.append(step)
            
        return valid_plan

    async def route_request(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Try LangChain planning first. If it fails or returns nothing, fall back to normal Compass-based routing.
        """
        try:
            plan = await self.plan_tool_chain(query, context)
        except Exception as exc:
            logger.warning(f"LangChain planning failed: {exc}")
            plan = []

        if not plan:
            logger.info("LangChain returned no plan — falling back to Compass routing.")
            return await super().route_request(query, context)

        # Check if we should execute a single tool or a chain
        if len(plan) == 1:
            # Single tool execution
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
        else:
            # Tool chain execution - convert all steps to endpoints
            endpoints = []
            for step in plan:
                endpoints.append({
                    "type": "tool",
                    "name": step["tool"],
                    "params": step.get("parameters", {}),
                })
                
            return {
                "intent": "tool_chain",
                "endpoints": endpoints,
                "confidence": 0.9,
                "prompt_type": "general",
                "plan": plan  # Include the full plan for reference
            }
            
    async def execute_tool(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute a tool with the given parameters."""
        if context is None:
            context = {}
            
        logger.info(f"Executing tool: {tool_name} with params: {params}")
        
        try:
            # If tool name contains namespace, split it
            if ":" in tool_name:
                namespace, name = tool_name.split(":", 1)
            else:
                namespace, name = "default", tool_name
                
            # Get the tool from MCP
            if self.mcp:
                tools = await self.mcp.get_tools()
                if name in tools:
                    from fastmcp import Context
                    
                    # Create context
                    ctx = Context(context or {})
                    
                    # Get the tool function
                    tool_func = tools[name].fn
                    
                    try:
                        # Try calling without context first (for @mcp.tool() decorated functions)
                        if params:
                            result = await tool_func(**params)
                        else:
                            result = await tool_func()
                    except TypeError:
                        # If TypeError, try with context
                        if params:
                            result = await tool_func(ctx, **params)
                        else:
                            result = await tool_func(ctx)
                            
                    return result
            
            # Fall back to super implementation
            return await super().execute_tool(tool_name, params, context)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"
