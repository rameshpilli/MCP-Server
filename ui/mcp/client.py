import os
import httpx
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable
from fastmcp import FastMCP, MCPRequest, MCPResponse, Tool, Context
from pathlib import Path
import sys
from .doc_reader import doc_reader

# Add the app directory to the Python path to access config
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.config import config
from app.utils.logger import log_interaction, log_error

class MCPClient:
    def __init__(self):
        self.cohere_index_name = config.COHERE_INDEX_NAME
        self.cohere_server_url = config.COHERE_SERVER_URL
        self.cohere_bearer_token = config.COHERE_SERVER_BEARER_TOKEN
        self.llm_model = config.LLM_MODEL
        self.llm_base_url = config.LLM_BASE_URL
        self.llm_oauth_endpoint = config.LLM_OAUTH_ENDPOINT
        self.llm_oauth_client_id = config.LLM_OAUTH_CLIENT_ID
        self.llm_oauth_client_secret = config.LLM_OAUTH_CLIENT_SECRET
        self.llm_oauth_grant_type = config.LLM_OAUTH_GRANT_TYPE
        self.llm_oauth_scope = config.LLM_OAUTH_SCOPE
        
        # MCP Server connection
        self.mcp_server_url = config.MCP_SERVER_URL if hasattr(config, 'MCP_SERVER_URL') else "http://localhost:8080"

        # Initialize FastMCP client
        self.mcp = FastMCP(
            model=self.llm_model,
            base_url=self.llm_base_url,
            oauth_endpoint=self.llm_oauth_endpoint,
            oauth_client_id=self.llm_oauth_client_id,
            oauth_client_secret=self.llm_oauth_client_secret,
            oauth_grant_type=self.llm_oauth_grant_type,
            oauth_scope=self.llm_oauth_scope
        )

    async def process_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message through the MCP pipeline using FastMCP
        """
        start_time = time.time()
        try:
            # Log incoming message
            log_interaction(
                step="receive_message",
                message=message,
                session_id=session_id
            )

            # Get context
            context = await self._get_context(message)
            context_found = bool(context.get("results", []))
            log_interaction(
                step="get_context",
                message=f"Context search completed",
                session_id=session_id,
                context_found=context_found
            )

            # Call the MCP server to process the message
            async with httpx.AsyncClient(timeout=60.0) as client:
                log_interaction(
                    step="mcp_server_request",
                    message=f"Sending request to MCP server: {self.mcp_server_url}",
                    session_id=session_id
                )
                
                # Prepare system message with context
                system_message = self._prepare_system_message(context)
                
                # Send request to MCP server
                response = await client.post(
                    f"{self.mcp_server_url}/process",
                    json={
                        "message": message,
                        "session_id": session_id,
                        "system_message": system_message,
                        "context": context
                    }
                )
                response.raise_for_status()
                server_response = response.json()
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Log successful response
                log_interaction(
                    step="mcp_process_complete",
                    message=server_response.get("message", ""),
                    session_id=session_id,
                    tools_used=server_response.get("tools_executed", []),
                    context_found=context_found,
                    response_length=len(server_response.get("message", "")),
                    processing_time_ms=processing_time
                )
                
                return {
                    "response": server_response.get("message", ""),
                    "tools_executed": server_response.get("tools_executed", []),
                    "context": context,
                    "processing_time_ms": processing_time
                }

        except Exception as e:
            # Log error
            log_error("process_message", e, session_id)
            raise Exception(f"Error processing message: {str(e)}")

    def _prepare_system_message(self, context: Dict[str, Any]) -> str:
        """
        Prepare system message with context and instructions
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
4. Use available tools when appropriate

Available tools:
- search_docs: Search through local documentation
- list_docs: List all available local documents
- read_doc: Read a specific document by name
- summarize_doc: Get a summary of a document"""

    async def _get_context(self, message: str) -> Dict[str, Any]:
        """
        Get relevant context for the message
        """
        try:
            if not self.cohere_server_url or not self.cohere_bearer_token:
                log_interaction(
                    step="get_context",
                    message="No Cohere configuration found",
                    context_found=False
                )
                return {"results": []}

            async with httpx.AsyncClient() as client:
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

        except Exception as e:
            log_error("get_context", e)
            return {"results": []}

# Create global client instance
mcp_client = MCPClient() 