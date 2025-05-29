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
            from langchain.agents import create_openai_functions_agent, create_react_agent
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
        self._create_openai_functions_agent = create_openai_functions_agent
        self._create_react_agent = create_react_agent

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
            
            # Try simple LLM planning first
            logger.info("Trying simple LLM planning approach...")
            simple_plan = await self._simple_plan_with_llm(query, tools, context)
            if simple_plan:
                logger.info(f"Simple LLM planning succeeded with {len(simple_plan)} steps")
                return simple_plan
            
            logger.info("Simple LLM planning returned no results, trying LangChain agent...")
                
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
            
            # Use a simpler agent approach without function binding
            # Convert tools to OpenAI functions
            # llm_with_tools = self.llm.bind(
            #     functions=[self._format_tool_to_openai_function(t) for t in tools]
            # )
            
            # Create the agent without function binding for now
            from langchain.agents import create_openai_functions_agent
            
            try:
                # Try the newer agent creation method
                agent = create_openai_functions_agent(
                    llm=self.llm,
                    tools=tools,
                    prompt=prompt
                )
            except Exception as e:
                logger.warning(f"Failed to create OpenAI functions agent: {e}")
                # Fall back to simpler agent
                from langchain.agents import create_react_agent
                
                # Modify prompt for ReAct agent
                react_prompt = self._ChatPromptTemplate.from_messages([
                    ("system", """You are an expert assistant that helps users by using tools when needed.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""),
                    ("human", "{input}"),
                    self._MessagesPlaceholder(variable_name="agent_scratchpad")
                ])
                
                agent = create_react_agent(
                    llm=self.llm,
                    tools=tools,
                    prompt=react_prompt
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
            # Tool chain execution - use sequential execution with context passing
            logger.info(f"Executing tool chain with {len(plan)} steps")
            
            # Execute the chain with context passing
            chain_results = await self._execute_tool_chain_with_context(plan, query)
            
            # Convert to the expected format
            endpoints = []
            for result in chain_results:
                endpoints.append({
                    "type": "tool",
                    "name": result["tool"],
                    "params": result.get("parameters", {}),
                    "result": result.get("result"),
                    "error": result.get("error")
                })
                
            return {
                "intent": "tool_chain",
                "endpoints": endpoints,
                "confidence": 0.9,
                "prompt_type": "general",
                "plan": plan,  # Include the original plan
                "chain_results": chain_results  # Include the execution results
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

    async def _simple_plan_with_llm(self, query: str, tools: List[Any], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Simple tool planning using direct LLM call instead of LangChain agents."""
        logger.info(f"Using simple LLM planning for query: '{query}'")
        
        # Create a simple prompt for tool planning
        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        tools_text = "\n".join(tool_descriptions)
        
        planning_prompt = f"""You are a helpful assistant that can use tools to answer user questions.

Available tools:
{tools_text}

User query: {query}

Please analyze the query and determine which tools (if any) should be used to answer it.
Respond with a JSON list of tool calls in this format:
[
  {{
    "tool": "tool_name",
    "parameters": {{"param1": "value1", "param2": "value2"}}
  }}
]

IMPORTANT RULES:
1. If no tools are needed, respond with an empty list: []
2. Only include tools that are directly relevant to answering the user's question
3. For tool chains where one tool's output feeds into another, use placeholders like:
   - "comma_separated_top_5_client_names" for client names from previous results
   - "previous_result" for the full output of the previous tool
4. When getting top clients and then analyzing them, the second tool should use the client names from the first
5. Always specify currency (USD or CAD) and region when relevant
6. Pay attention to time period requests:
   - "previous year" or "last year" → include context to use RevenuePrevYear
   - "previous YTD" → include context to use RevenuePrevYTD  
   - default → use RevenueYTD

For the query "{query}", what tools should be called?"""

        try:
            # Use our LLM directly
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=planning_prompt)]
            
            result = await self.llm.ainvoke(messages)
            response_text = result.content
            
            logger.debug(f"LLM planning response: {response_text}")
            
            # Try to parse JSON from the response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    plan = json.loads(json_str)
                    return self._validate_plan(plan)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from LLM response: {json_str}")
            
            return []
        except Exception as e:
            logger.error(f"Simple LLM planning failed: {e}")
            return []

    async def _execute_tool_chain_with_context(self, plan: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Execute a tool chain where each tool can use the output of previous tools."""
        results = []
        context_data = {}
        
        for i, step in enumerate(plan):
            tool_name = step["tool"]
            params = step.get("parameters", {})
            
            logger.info(f"Executing tool {i+1}/{len(plan)}: {tool_name}")
            
            # Replace parameter placeholders with actual data from previous results
            if i > 0:  # Not the first tool
                params = await self._resolve_parameter_placeholders(params, results, context_data)
            
            try:
                # Execute the tool
                result = await self.execute_tool(tool_name, params)
                
                # Store result for future tools
                step_result = {
                    "tool": tool_name,
                    "parameters": params,
                    "result": result,
                    "step": i + 1
                }
                results.append(step_result)
                
                # Extract useful data for next tools
                context_data[f"tool_{i}_result"] = result
                context_data[f"tool_{i}_name"] = tool_name
                
                # Try to extract specific data types for common use cases
                if "client" in tool_name.lower() and isinstance(result, str):
                    # Try to extract client names from table results
                    client_names = self._extract_client_names_from_result(result)
                    if client_names:
                        context_data["extracted_client_names"] = client_names
                        context_data["client_names_csv"] = ",".join(client_names)
                
                logger.info(f"Tool {tool_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                step_result = {
                    "tool": tool_name,
                    "parameters": params,
                    "error": str(e),
                    "step": i + 1
                }
                results.append(step_result)
                # Continue with remaining tools even if one fails
        
        return results
    
    async def _resolve_parameter_placeholders(self, params: Dict[str, Any], previous_results: List[Dict[str, Any]], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Replace placeholder values in parameters with actual data from previous tool results."""
        resolved_params = params.copy()
        
        for key, value in resolved_params.items():
            if isinstance(value, str):
                # Replace common placeholders
                if "comma_separated_top_5_client_names" in value or "client_names" in key.lower():
                    if "client_names_csv" in context_data:
                        resolved_params[key] = context_data["client_names_csv"]
                        logger.debug(f"Resolved {key}: {value} -> {context_data['client_names_csv']}")
                
                # Replace other placeholders with previous results
                if "previous_result" in value:
                    if previous_results:
                        resolved_params[key] = previous_results[-1]["result"]
                        logger.debug(f"Resolved {key} with previous result")
        
        return resolved_params
    
    def _extract_client_names_from_result(self, result: str) -> List[str]:
        """Extract client names from a table result."""
        try:
            import re
            # Look for markdown table format
            lines = result.split('\n')
            client_names = []
            
            for line in lines:
                # Skip header and separator lines
                if '|' in line and not line.strip().startswith('|:') and 'Client Name' not in line:
                    parts = line.split('|')
                    if len(parts) > 1:
                        # First column after | is usually the client name
                        client_name = parts[1].strip()
                        if client_name and client_name not in ['---', '']:
                            client_names.append(client_name)
            
            # Remove duplicates and limit
            client_names = list(dict.fromkeys(client_names))[:5]
            logger.debug(f"Extracted client names: {client_names}")
            return client_names
            
        except Exception as e:
            logger.warning(f"Failed to extract client names: {e}")
            return []
