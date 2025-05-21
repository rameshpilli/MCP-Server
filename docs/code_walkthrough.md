# Code Walkthrough

This guide provides a human friendly overview of the repository and where to find the most important pieces of code.  Each section lists the key files and highlights notable functions.

## Repository Structure

- `run.py` – launcher that starts the MCP server and the FastAPI API.
- `mcp_client.py` – lightweight Python SDK for sending queries to the server.
- `client_cli.py` – small CLI wrapper around `MCPClient`.
- `app/` – main application code.
  - `config.py` – central configuration.
  - `main.py` – FastAPI entry point.
  - `mcp_server.py` – core MCP server logic and tool registry.
  - `mcp_bridge.py` – routes queries to tools and coordinates execution.
  - `client.py` – helper for calling LLM APIs.
  - `utils/` – collection of helper modules.
  - `registry/` – tool registration utilities.
  - `prompts/`, `memory/`, `resources/`, etc. – supporting components.

## Important Modules

### `app/mcp_server.py`
Handles FastAPI endpoints and coordinates tool execution.  Tools are discovered at start up using `registry.autodiscover_tools`.

Key functions:
- `process_message` – orchestrates tool execution for a user message.
- `main_factory` – builds the FastAPI application with all routes.

### `app/mcp_bridge.py`
Implements the high level planning logic.  It can generate tool chains using an LLM and executes them through `ResultsCoordinator`.

Key functions:
- `plan_tool_chain` – ask the LLM to create a list of tool calls.
- `run_plan`/`run_plan_streaming` – execute a plan synchronously or as a stream.
- `_validate_plan` – checks each step against the `PlanStep` schema before running it.

### `app/client.py`
Provides a simple wrapper to call the configured language model.

Key function:
- `call_llm` – sends chat messages to the LLM with proper headers and returns the JSON response.

### `app/utils/parameter_extractor.py`
Extracts tool parameters from natural language queries.

Key functions:
- `_call_llm` – minimal LLM wrapper used only for parameter extraction.
- `extract_parameters_with_llm` – high level routine that formats the prompt and normalises the output.

### `app/registry/tools.py`
Manages tool registration.  Tools are defined with the `@register_tool` decorator and stored in a registry.

Key functions:
- `register_tool` – decorator to register a tool handler.
- `autodiscover_tools` – imports modules from `app.tools` and calls their `register_tools` function if present.

### `mcp_client.py` and `client_cli.py`
The SDK (`MCPClient`) allows other Python code to query the MCP server.  The CLI script exposes the same functionality on the command line.

## LLM Wrapper Functions
The repository has several small functions that directly interact with the LLM service:

1. `app/utils/parameter_extractor.py:_call_llm` – used when extracting parameters for a tool.
2. `app/client.py:call_llm` – the main wrapper for MCP server interactions.
3. `ui/app.py:call_llm` – similar wrapper used by the Chainlit UI.

These wrappers all construct an HTTP request to the LLM endpoint, set authentication headers (OAuth token or API key), and return the parsed JSON result.

## Step‑by‑Step Example
1. A user query arrives at `app/main.py` through `/api/v1/chat`.
2. `main.py` calls `mcp_bridge.MCPBridge.route_request` to find the right tool.
3. `MCPBridge` may call `plan_tool_chain` to let the LLM decide a multi tool plan.
4. Each planned step is validated by `_validate_plan` and executed via `run_plan`.
5. Individual tools are registered through decorators in `app/tools/*` and executed by `app/mcp_server.py`.
6. Results flow back up to the API response or the Chainlit UI.

