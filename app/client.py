# app/client.py
import os
import httpx
import time
import asyncio
import socket
from typing import Optional, Dict, Any, List
from .config import config
from .utils.logger import log_interaction, log_error
from .mcp_bridge import MCPBridge
from fastmcp import Context
import logging

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        # Configuration only - no mutable state for horizontal scaling
        self.cohere_index_name = config.COHERE_INDEX_NAME
        self.cohere_server_url = config.COHERE_SERVER_URL
        self.cohere_bearer_token = config.COHERE_SERVER_BEARER_TOKEN
        self.llm_model = config.LLM_MODEL
        self.llm_base_url = config.LLM_BASE_URL
        
        # Initialize the MCP Bridge - stateless component
        self.bridge = MCPBridge()
        
        # Cache timeouts for connection discovery
        self._mcp_port_cache = None
        self._mcp_port_cache_time = 0
        self._mcp_port_cache_ttl = 60  # seconds

    # Gets OAuth token:
    async def _get_oauth_token(self) -> str:
        """
        Get OAuth token for LLM authentication
        
        This method:
        1. Calls the OAuth endpoint configured in .env
        2. Gets an access token using client credentials
        3. Returns the token for use in LLM API calls
        """
        try:
            oauth_endpoint = config.LLM_OAUTH_ENDPOINT
            client_id = config.LLM_OAUTH_CLIENT_ID
            client_secret = config.LLM_OAUTH_CLIENT_SECRET
            
            if not all([oauth_endpoint, client_id, client_secret]):
                # If OAuth is not configured, return empty string
                return ""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    oauth_endpoint,
                    data={
                        "grant_type": config.LLM_OAUTH_GRANT_TYPE or "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": config.LLM_OAUTH_SCOPE or "read"
                    }
                )
                response.raise_for_status()
                
                token_data = response.json()
                return token_data.get("access_token", "")
                
        except Exception as e:
            logger.warning(f"Failed to get OAuth token: {e}")
            return ""
    
    async def _discover_mcp_server_port(self):
        """
        Detect the actual MCP server port by checking ports around the configured port
        
        In Kubernetes, we use service discovery instead of port scanning
        
        Returns:
            The port where MCP server is running or None if not found
        """
        # In Kubernetes, use the configured port (service discovery)
        if config.IN_KUBERNETES:
            return config.MCP_SERVER_PORT
            
        # Check cache first
        current_time = time.time()
        if self._mcp_port_cache and (current_time - self._mcp_port_cache_time < self._mcp_port_cache_ttl):
            return self._mcp_port_cache
        
        start_port = config.MCP_SERVER_PORT
        logger.info(f"Attempting to discover MCP server starting at port {start_port}")
        
        # Check 10 ports starting from the configured port
        for port_offset in range(10):
            port = start_port + port_offset
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect((config.MCP_SERVER_HOST, port))
                    logger.info(f"Found MCP server running on port {port}")
                    
                    # Update cache
                    self._mcp_port_cache = port
                    self._mcp_port_cache_time = current_time
                    
                    return port
            except (ConnectionRefusedError, socket.timeout):
                continue
        
        logger.warning("Could not find a running MCP server")
        return None
 
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the configured LLM using OpenAI API
        
        Stateless method that doesn't depend on client instance state
        """
        logger.info("_call_llm called")
        try:
            # Get OAuth token first (if configured)
            token = await self._get_oauth_token()
            logger.info(f"Got OAuth token: {'Yes' if token else 'No'}")
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add token if available, or use API key from environment
            if token:
                headers["Authorization"] = f"Bearer {token}"
            elif os.getenv('OPENAI_API_KEY'):
                headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
                logger.info("Using OPENAI_API_KEY from environment")
            
            # Call the LLM with proper error handling and timeouts
            logger.info(f"Calling LLM at {self.llm_base_url}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "temperature": 0.7
                    },
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"LLM returned: {content[:100]}...")
                return content
            
        except httpx.ReadTimeout:
            logger.error("LLM request timed out")
            return "I'm sorry, but the response is taking longer than expected. Please try again with a simpler question."
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in _call_llm: {e.response.status_code} - {e.response.text}")
            log_error("call_llm_http", e)
            return "I'm experiencing connection issues with my knowledge service. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Error in _call_llm: {e}")
            log_error("call_llm", e)
            return "I understand your question, but I'm having trouble processing it right now. Please try again."

    async def process_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message through the MCP pipeline using the bridge
        
        Stateless method suitable for horizontal scaling
        """
        start_time = time.time()
        request_id = f"{int(time.time() * 1000)}-{hash(message) % 10000}"
        
        try:
            # Log incoming message with request ID for tracing
            log_interaction(
                step="receive_message",
                message=f"[{request_id}] {message}",
                session_id=session_id
            )

            # Get context from Cohere with proper error handling
            try:
                context = await self._get_context(message)
                context_found = bool(context.get("results", []))
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                context = {"results": []}
                context_found = False
            
            log_interaction(
                step="get_context",
                message=f"[{request_id}] Context search completed",
                session_id=session_id,
                context_found=context_found
            )

            # Use the bridge to route the request
            routing = await self.bridge.route_request(message, context)
            
            # Execute tools based on routing
            results = []
            tools_executed = []
            
            # Discover actual MCP server port if needed
            actual_port = await self._discover_mcp_server_port()
            if actual_port and actual_port != config.MCP_SERVER_PORT:
                logger.info(f"Discovered MCP server on port {actual_port}, updating config")
                config.MCP_SERVER_PORT = actual_port
            
            # Execute tools with retry logic for resilience
            for endpoint in routing["endpoints"]:
                if endpoint["type"] == "tool":
                    tool_name = endpoint["name"]
                    params = endpoint.get("params", {})
                    
                    # Try to execute the tool with retries
                    for attempt in range(3):  # Up to 3 attempts
                        try:
                            # Execute tool through the MCP server
                            from .mcp_server import mcp
                            
                            # Create a mock context for the tool
                            from .registry import tool_registry, resource_registry, prompt_registry
                            
                            class MockLifespanContext:
                                def __init__(self):
                                    self.tool_registry = tool_registry
                                    self.resource_registry = resource_registry
                                    self.prompt_registry = prompt_registry
                            
                            class MockRequestContext:
                                def __init__(self):
                                    self.lifespan_context = MockLifespanContext()
                                    self.context = {}
                                    # Add request trace info
                                    self.request_id = request_id
                                    self.session_id = session_id
                            
                            class MockContext:
                                def __init__(self):
                                    self.request_context = MockRequestContext()
                            
                            ctx = MockContext()
                            
                            # Get all tools from the MCP server
                            tools = await mcp.get_tools()
                            
                            # Find the specific tool we want
                            tool = tools.get(tool_name)
                            
                            if tool:
                                # Get the actual function from the tool object
                                tool_func = tool.fn
                                
                                # Check the tool signature
                                import inspect
                                sig = inspect.signature(tool_func)
                                param_names = list(sig.parameters.keys())
                                
                                # Execute with appropriate parameters
                                if len(param_names) > 1:
                                    result = await tool_func(ctx, **params)
                                else:
                                    result = await tool_func(ctx)
                                
                                results.append(result)
                                tools_executed.append(tool_name)
                                
                                # Tool executed successfully, break the retry loop
                                break
                            else:
                                logger.warning(f"Tool '{tool_name}' not found")
                                results.append(f"Tool '{tool_name}' not found")
                                break  # No need to retry for missing tools
                                
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name} (attempt {attempt+1}): {e}")
                            
                            if attempt == 2:  # Last attempt
                                # Add error message to results
                                results.append(f"Error executing tool '{tool_name}': {str(e)}")
                            else:
                                # Wait before retrying
                                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

            # Combine results
            response = await self._combine_results(results, routing["intent"], message, context)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful response
            log_interaction(
                step="mcp_process_complete",
                message=f"[{request_id}] {response[:100]}...",
                session_id=session_id,
                tools_used=tools_executed,
                context_found=context_found,
                response_length=len(response),
                processing_time_ms=processing_time
            )

            return {
                "response": response,
                "tools_executed": tools_executed,
                "context": context,
                "processing_time_ms": processing_time,
                "request_id": request_id
            }

        except Exception as e:
            # Log error and return graceful error message
            log_error(f"process_message_{request_id}", e, session_id)
            return {
                "response": f"I encountered an error while processing your request. Error details: {str(e)}",
                "tools_executed": [],
                "context": {},
                "processing_time_ms": (time.time() - start_time) * 1000,
                "request_id": request_id,
                "error": str(e)
            }

    def _prepare_system_message(self, context: Dict[str, Any]) -> str:
        """
        Prepare system message with context and instructions
        
        Pure function with no side effects or network calls
        """
        # Extract relevant information from context
        context_info = ""
        if context.get("results"):
            context_info = "\nRelevant context:\n" + "\n".join(
                [f"- {result['text']}" for result in context["results"]]
            )

        return f"""You are an AI assistant powered by {self.llm_model}. 
Your role is to help users by providing accurate and helpful responses.
{context_info}

Please follow these guidelines:
1. Use the provided context when relevant
2. Be concise but thorough
3. If you're unsure, say so
4. Use available tools when appropriate"""

    async def _get_context(self, message: str) -> Dict[str, Any]:
        """
        Get relevant context for the message from Compass
        
        This method tries to find relevant information from the Cohere Compass index
        based on the user's query. It uses semantic search to find the most relevant
        content that can help answer the query.
        
        Uses both library and API approaches with proper fallbacks
        """
        try:
            if not self.cohere_server_url or not self.cohere_bearer_token:
                log_interaction(
                    step="get_context",
                    message="No Cohere configuration found",
                    context_found=False
                )
                return {"results": []}

            # First try the library approach
            try:
                from cohere_compass.clients.compass import CompassClient
                import asyncio
                
                # Create the Compass client
                client = CompassClient(
                    index_url=self.cohere_server_url,
                    bearer_token=self.cohere_bearer_token
                )
                
                # Run the search in a thread to avoid blocking
                search_result = await asyncio.to_thread(
                    lambda: client.search_chunks(
                        index_name=self.cohere_index_name,
                        query=message,
                        top_k=5
                    )
                )
                
                # Process the results into the expected format
                results = []
                if hasattr(search_result, 'hits'):
                    for hit in search_result.hits:
                        # Extract content text
                        content_text = ""
                        if hasattr(hit, 'content'):
                            if isinstance(hit.content, dict):
                                content_text = hit.content.get('text', '')
                            else:
                                content_text = str(hit.content)
                        
                        # Extract score
                        score = 0.0
                        if hasattr(hit, 'score'):
                            score = hit.score
                        
                        results.append({
                            "text": content_text,
                            "score": score
                        })
                
                logger.info(f"Found {len(results)} relevant items in Compass")
                return {"results": results}
                
            except ImportError as e:
                # Fall back to direct API call if library not available
                logger.warning(f"Library approach failed: {e}, falling back to API call")
                
                # Use httpx with timeout and retry logic
                for attempt in range(3):  # Up to 3 attempts
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.post(
                                f"{self.cohere_server_url}/search",
                                headers={"Authorization": f"Bearer {self.cohere_bearer_token}"},
                                json={
                                    "index_name": self.cohere_index_name,
                                    "query": message,
                                    "max_results": 5
                                }
                            )
                            response.raise_for_status()
                            return response.json()
                    except httpx.ReadTimeout:
                        logger.warning(f"Compass search timed out (attempt {attempt+1})")
                        if attempt == 2:  # Last attempt
                            return {"results": []}
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    except Exception as e:
                        logger.error(f"Error in API-based context search (attempt {attempt+1}): {e}")
                        if attempt == 2:  # Last attempt
                            return {"results": []}
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        except Exception as e:
            logger.error(f"Error getting context: {e}")
            log_error("get_context", e)
            return {"results": []}

    async def _combine_results(self, results: List[Any], intent: str, original_query: str, context: Dict[str, Any]) -> str:
        """
        Combine results into a response using LLM
        
        Stateless method suitable for horizontal scaling
        """
        logger.info(f"=== _combine_results START ===")
        logger.info(f"Intent: {intent}")
        
        if not results:
            return "I couldn't find any relevant information for your request."
        
        # Check what exactly is in results
        for i, result in enumerate(results):
            logger.info(f"Result {i}: Type={type(result)}, Content={str(result)[:100]}...")
        
        # Prepare results text with protection against very long results
        results_text = ""
        for result in results:
            # Limit each result to a reasonable size to avoid token limits
            result_str = str(result)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "... (truncated for brevity)"
            results_text += result_str + "\n\n"
        
        # Create messages for LLM with clear instructions
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant. Use the provided search results to answer the user's question.
                
                The user asked: "{original_query}"
                
                Here are the search results:
                {results_text}
                
                Please provide a natural, conversational answer to the user's question based on these search results.
                If the search results don't contain the information needed, acknowledge this rather than making up information."""
            },
            {
                "role": "user",
                "content": original_query
            }
        ]
        
        # Call LLM with retry logic
        for attempt in range(3):  # Up to 3 attempts
            try:
                llm_response = await self._call_llm(messages)
                logger.info(f"LLM response received: {llm_response[:100]}...")
                return llm_response
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt+1}): {e}", exc_info=True)
                if attempt == 2:  # Last attempt
                    # Fallback to basic response
                    return f"Based on your {intent} request, I found some information but had trouble processing it. Here's what I found:\n\n{results_text[:500]}..."
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

    # Add a health check method for Kubernetes readiness probes
    async def check_mcp_server_connection(self) -> bool:
        """
        Check if the MCP server is accessible
        
        Used by health checks and readiness probes
        
        Returns:
            bool: True if MCP server is accessible, False otherwise
        """
        try:
            port = await self._discover_mcp_server_port()
            return port is not None
        except Exception as e:
            logger.error(f"Error checking MCP server connection: {e}")
            return False

#  Global client instance
mcp_client = MCPClient()