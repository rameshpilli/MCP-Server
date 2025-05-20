"""
MCP Server Module

This module provides the main MCP server implementation.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterable
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import importlib
from fastmcp import FastMCP, Context
from typing import Any, Dict, Optional
from app.registry.tools import registry as tool_registry
from app.registry.resources import registry as resource_registry
from app.registry.prompts import registry as prompt_registry
from app.config import config
from app.utils.port_utils import find_available_port
from app.utils.logging import setup_logging

# Add doc_reader to path
sys.path.append(str(Path(__file__).parent.parent))
from app.config import config

# Setup logging using centralized configuration
logger = setup_logging("mcp_server")
logger.info("MCP server starting...")

# Load environment variables
load_dotenv()

# Import registries after loading environment variables
from app.registry import tool_registry, resource_registry, prompt_registry
import app.tools
import app.resources
import app.prompts

@dataclass
class ServerContext:
    """Context for the MCP Server"""
    # Registries
    tool_registry: tool_registry.__class__
    resource_registry: resource_registry.__class__
    prompt_registry: prompt_registry.__class__
    bridge: Any  # Will be set to MCPBridge instance

@asynccontextmanager
async def mcp_server_lifespan(server: FastMCP) -> AsyncIterable[ServerContext]:
    """
    Manage the MCP Server lifespan.

    Args:
        server: The FastMCP server instance
    Yields:
        ServerContext: The context for the MCP server.
    """
    logger.info('Starting MCP Server')
    
    try:
        # Signal that we're starting up - useful for readiness probes
        if config.IN_KUBERNETES:
            # Create a startup marker that could be used by health checks
            startup_file = '/tmp/mcp_server_ready'
            with open(startup_file, 'w') as f:
                f.write('ready')
            logger.info(f"Created startup marker at {startup_file}")
        
        # Import MCPBridge here to avoid circular import
        from app.mcp_bridge import MCPBridge
        bridge = MCPBridge()
        logger.info("MCPBridge initialized successfully")
            
        # Initialize any resources needed for the server lifetime
        yield ServerContext(
            tool_registry=tool_registry,
            resource_registry=resource_registry,
            prompt_registry=prompt_registry,
            bridge=bridge
        )
    except Exception as e:
        logger.error(f"Error during MCP server startup: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up resources
        logger.info('Shutting down MCP Server')
        
        # Remove startup marker if it exists
        if config.IN_KUBERNETES:
            startup_file = '/tmp/mcp_server_ready'
            if os.path.exists(startup_file):
                os.remove(startup_file)
                logger.info(f"Removed startup marker at {startup_file}")

# Create the MCP server
mcp = FastMCP(
    config.SERVER_NAME,
    description=config.SERVER_DESCRIPTION,
    lifespan=mcp_server_lifespan,
    host=config.MCP_SERVER_HOST if hasattr(config, 'MCP_SERVER_HOST') else "localhost",
    port=config.MCP_SERVER_PORT if hasattr(config, 'MCP_SERVER_PORT') else "8080"
)

# Register message handler
async def process_message(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Process incoming messages using the MCPBridge for tool execution and response generation.
    """
    try:
        # Create a Context object
        ctx = Context(context or {})
        
        # Log the incoming message
        logger.info(f"Step: message_received | Message: {message} | Session: {ctx.session_id} | Status: processing")

        # Route the request using the bridge
        routing_result = await ctx.bridge.route_request(message, ctx.context)

        # Execute tools based on routing result
        results = []
        for endpoint in routing_result["endpoints"]:
            if endpoint["type"] == "tool":
                try:
                    logger.info(f"Executing tool: {endpoint['name']}")
                    result = await ctx.bridge.mcp.execute_tool(
                        endpoint["name"],
                        endpoint.get("params", {}),
                        ctx.context
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error in tool_execution_{endpoint['name']} | Error: {e} | Session: {ctx.session_id}")

        # Format response based on results
        if results:
            response_parts = ["Based on the tool execution results:"]
            for i, result in enumerate(results, 1):
                response_parts.append(f"{i}. {result}")
            response_text = "\n".join(response_parts)
        else:
            response_text = "I've processed your request but didn't find any specific results to share."

        # Log the response
        logger.info(
            f"Step: mcp_process_complete | Message: {response_text[:100]}... | Session: {ctx.session_id} | Status: success")

        return response_text

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

# Set the process method
mcp.process = process_message

def register_all_components():
    """Register all components (tools, integrations, resources, and prompts)"""
    logger.info("Registering MCP components...")

    # 1. Register tools from tools directory
    tools_dir = Path(__file__).parent / "tools"
    if not tools_dir.exists():
        logger.warning(f"Tools directory not found: {tools_dir}")
    else:
        # Register tools from each module
        tool_modules = [f.stem for f in tools_dir.glob("*.py") if f.stem != "__init__"]
        for module_name in tool_modules:
            try:
                # Import the module
                module = importlib.import_module(f"app.tools.{module_name}")

                # Register tools if the module has a register_tools function
                if hasattr(module, "register_tools"):
                    logger.info(f"Registering tools from {module_name}")
                    module.register_tools(mcp)
                else:
                    logger.warning(f"Module {module_name} has no register_tools function")
            except Exception as e:
                logger.error(f"Error registering tools from {module_name}: {e}")

    # 2. Register tools from integrations directory
    integrations_dir = Path(__file__).parent / "integrations"
    if not integrations_dir.exists():
        logger.warning(f"Integrations directory not found: {integrations_dir}")
    else:
        # Register tools from each integration module
        integration_modules = [f.stem for f in integrations_dir.glob("*.py") if f.stem != "__init__"]
        for module_name in integration_modules:
            try:
                # Import the module
                module = importlib.import_module(f"app.integrations.{module_name}")

                # Register tools if the module has a register_tools function
                if hasattr(module, "register_tools"):
                    logger.info(f"Registering integration tools from {module_name}")
                    module.register_tools(mcp)
                else:
                    logger.info(f"Integration module {module_name} has no register_tools function")
            except Exception as e:
                logger.error(f"Error registering integration tools from {module_name}: {e}")

    # 3. Register resources from resources directory
    resources_dir = Path(__file__).parent / "resources"
    if not resources_dir.exists():
        logger.warning(f"Resources directory not found: {resources_dir}")
    else:
        # Register resources from each module
        resource_modules = [f.stem for f in resources_dir.glob("*.py") if f.stem != "__init__"]
        for module_name in resource_modules:
            try:
                # Import the module
                module = importlib.import_module(f"app.resources.{module_name}")

                # Register resources if the module has a register_resources function
                if hasattr(module, "register_resources"):
                    logger.info(f"Registering resources from {module_name}")
                    module.register_resources(mcp)
                else:
                    logger.warning(f"Module {module_name} has no register_resources function")
            except Exception as e:
                logger.error(f"Error registering resources from {module_name}: {e}")

    # 4. Register prompts from prompts directory
    prompts_dir = Path(__file__).parent / "prompts"
    if not prompts_dir.exists():
        logger.warning(f"Prompts directory not found: {prompts_dir}")
    else:
        # Register prompts from each module
        prompt_modules = [f.stem for f in prompts_dir.glob("*.py") if f.stem != "__init__"]
        for module_name in prompt_modules:
            try:
                # Import the module
                module = importlib.import_module(f"app.prompts.{module_name}")

                # Register prompts if the module has a register_prompts function
                if hasattr(module, "register_prompts"):
                    logger.info(f"Registering prompts from {module_name}")
                    module.register_prompts(mcp)
                else:
                    logger.warning(f"Module {module_name} has no register_prompts function")
            except Exception as e:
                logger.error(f"Error registering prompts from {module_name}: {e}")

    # 5. Register agent components from agent directory
    agent_dir = Path(__file__).parent / "agent"
    if not agent_dir.exists():
        logger.warning(f"Agent directory not found: {agent_dir}")
    else:
        # Register agent components from each module
        agent_modules = [f.stem for f in agent_dir.glob("*.py") if f.stem != "__init__"]
        for module_name in agent_modules:
            try:
                # Import the module
                module = importlib.import_module(f"app.agent.{module_name}")

                # Register agent components if the module has a register_components function
                if hasattr(module, "register_components"):
                    logger.info(f"Registering agent components from {module_name}")
                    module.register_components(mcp)
                else:
                    logger.info(f"Agent module {module_name} has no register_components function")
            except Exception as e:
                logger.error(f"Error registering agent components from {module_name}: {e}")

    logger.info("Component registration complete")


# Server info tool with K8s awareness - useful for debugging and health checks
@mcp.tool()
async def server_info(ctx: Context, namespace: str = None) -> str:
    """Get information about the MCP server and available components."""
    try:
        # Get registries from context
        tool_reg = ctx.request_context.lifespan_context.tool_registry
        resource_reg = ctx.request_context.lifespan_context.resource_registry
        prompt_reg = ctx.request_context.lifespan_context.prompt_registry

        # Get components
        tools = tool_reg.list_tools(namespace) if hasattr(tool_reg, 'list_tools') else {}
        resources = resource_reg.list_resources(namespace) if hasattr(resource_reg, 'list_resources') else {}
        prompts = prompt_reg.list_prompts(namespace) if hasattr(prompt_reg, 'list_prompts') else {}

        # Format tool list with proper escaping
        tool_list = []
        for name, desc in tools.items():
            # Escape the colon in markdown
            escaped_name = name.replace(':', '\\:')
            tool_list.append(f"- **{escaped_name}**: {desc}")
        tool_text = "\n".join(tool_list) if tool_list else "No tools registered"

        # Format resource list
        resource_list = []
        for name, desc in resources.items():
            escaped_name = name.replace(':', '\\:')
            resource_list.append(f"- **{escaped_name}**: {desc}")
        resource_text = "\n".join(resource_list) if resource_list else "No resources registered"

        # Format prompt list
        prompt_list = []
        for name, desc in prompts.items():
            escaped_name = name.replace(':', '\\:')
            prompt_list.append(f"- **{escaped_name}**: {desc}")
        prompt_text = "\n".join(prompt_list) if prompt_list else "No prompt templates registered"

        # Add environment info for debugging
        environment_info = "Local Development"
        if config.IN_KUBERNETES:
            environment_info = f"Kubernetes (Pod: {config.POD_NAME}, Namespace: {config.NAMESPACE})"

        return f"""
# MCP Server Information

**Environment**: {environment_info}
**Host**: {config.MCP_SERVER_HOST}
**Port**: {config.MCP_SERVER_PORT}

## Available Components

### Tools
{tool_text}

### Resources
{resource_text}

### Prompt Templates
{prompt_text}
"""
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        return f"Error getting server info: {e}"


# Register all components
register_all_components()


# Add health check endpoint - critical for Kubernetes
@mcp.tool()
async def health_check(ctx: Context) -> str:
    """
    Health check endpoint for Kubernetes liveness and readiness probes.

    This tool allows Kubernetes to monitor the health of the MCP server.
    """
    # Simply returning a success message is enough for a basic health check
    # In a more advanced implementation, you might check dependencies too
    return "MCP Server is healthy"


async def main():
    """Run the MCP server"""
    host = config.MCP_SERVER_HOST if hasattr(config, 'MCP_SERVER_HOST') else "localhost"
    configured_port = config.MCP_SERVER_PORT if hasattr(config, 'MCP_SERVER_PORT') else 8080

    # In Kubernetes, we use the configured port directly
    if config.IN_KUBERNETES:
        available_port = configured_port
        logger.info(f"Running in Kubernetes, using assigned port {available_port}")
    else:
        # For local development, find an available port
        available_port = find_available_port(configured_port)
        if not available_port:
            logger.error(
                f"Could not find an available port after trying {configured_port} through {configured_port + 9}")
            return

        # If we had to use a different port, log it
        if available_port != configured_port:
            logger.warning(f"Port {configured_port} was not available. Using port {available_port} instead.")

            # Update the config.MCP_SERVER_PORT to the new port
            config.MCP_SERVER_PORT = available_port

    logger.info(f"Starting MCP server on {host}:{available_port}")

    # In Kubernetes, always use SSE transport - better for containers
    transport = "sse" if config.IN_KUBERNETES else os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        await mcp.run_sse_async(host=host, port=available_port)
    else:
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())