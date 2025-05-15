# MCP Server Module Documentation

## Core Modules

### `app/main.py`
- FastAPI application server that provides HTTP endpoints for the MCP platform
- Implements chat, agent registration, and tool execution endpoints
- Handles CORS and request routing

### `app/mcp_server.py`
- Main MCP server implementation
- Manages tool registration and component initialization
- Handles message processing and routing
- Integrates with various registries (tools, resources, prompts)

### `app/mcp_bridge.py`
- Bridge between MCP server and external tools
- Provides request routing and tool execution capabilities
- Handles context management and error handling

### `app/client.py`
- MCP client implementation for interacting with the server
- Provides methods for tool execution and request handling
- Manages client-side state and connections

### `app/config.py`
- Configuration management for the entire application
- Handles environment variables and default settings
- Manages server, LLM, OAuth, and logging configurations

## Agent System

### `app/agent/`
- `__init__.py` - Module initialization file that exports the Agent and AgentRegistry classes
- `registration.py` - Implementation of the Agent and AgentRegistry classes
- `api.py` - FastAPI endpoints for agent registration and management

### `app/agents/`
- Contains agent implementations and management utilities
- Handles agent capabilities and registration

## Memory Management

### `app/memory/`
- `__init__.py` - Module initialization file for memory management
- `short_term.py` - Implementation of short-term memory using Redis with fallback to in-memory storage

## Tools and Resources

### `app/tools/`
- `__init__.py` - Package initialization that imports all tool modules
- Contains various tool implementations:
  - `document.py` - Document processing tools
  - `summarization.py` - Text summarization tools
  - `search.py` - Search functionality tools
  - `reporting.py` - Report generation tools
  - `analysis.py` - Data analysis tools

### `app/resources/`
- `__init__.py` - Package initialization for external resources
- `apis.py` - External API integrations and resource management

## Registry System

### `app/registry/`
- `__init__.py` - Exports tool, resource, and prompt registries
- `tools.py` - Tool registration and management
- `resources.py` - Resource registration and management
- `prompts.py` - Prompt template registration and management

## Prompts

### `app/prompts/`
- `__init__.py` - Package initialization for prompt templates
- `templates.py` - Prompt template definitions and management

## Utilities

### `app/utils/`
- `__init__.py` - Exports utility functions
- `port_utils.py` - Port management utilities
- `logging_config.py` - Logging configuration and setup
- `logger.py` - Advanced logging with CSV and S3 support
- `mcp_inspector.py` - MCP server inspection utilities

## Workers

### `app/workers/`
- Background task processing and worker management
- Handles asynchronous operations and task queues

## Testing

### `tests/`
- Contains various test modules for different components:
  - `check_compass_fields.py` - Tests CompassDocument model fields
  - `check_mcp_tools.py` - Tests MCP tool registration
  - `direct_tool_test.py` - Tests direct tool execution
  - `list_tools.py` - Tests tool listing functionality
  - `test_clientview_financials.py` - Tests financial data tools
  - `test_compass_index_*.py` - Tests Compass indexing functionality
  - `test_compass_search.py` - Tests search capabilities
  - `test_mcp_tools_summarize.py` - Tests summarization tools

## Configuration Files

### Root Level
- `Dockerfile` - Container configuration for the application
- `run.py` - Main launcher script for MCP and FastAPI servers
- `.chainlit/config.toml` - Chainlit chat interface configuration

## Key Features
1. Modular architecture with clear separation of concerns
2. Comprehensive tool registration and management system
3. Flexible agent system with registration and routing capabilities
4. Robust memory management with Redis integration
5. Extensive logging and monitoring capabilities
6. Container-ready with Kubernetes support
7. Secure configuration management
8. Comprehensive testing suite 