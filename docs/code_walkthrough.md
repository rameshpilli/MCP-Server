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

## Query Lifecycle Graph

The diagram below shows how a request moves through the main modules when hitting
the `/api/v1/chat` endpoint.  Arrows represent function calls and data flow.

```text
main.py (/api/v1/chat)
  └── MCPClient.process_message (app/client.py)
        ├── _get_context → call_llm (LLM for context search)
        └── MCPBridge.route_request (app/mcp_bridge.py)
              ├── plan_tool_chain → run_plan
              │       └── ResultsCoordinator.process_plan
              │             └── MCPBridge.execute_tool
              │                   └── mcp_server.process_message
              │                         └── execute_tool
              │                               └── registry.autodiscover_tools → tool_function
              └── _get_tool_params → extract_parameters_with_llm (LLM)
        └── _combine_results → call_llm (formats final answer)
  Response returned by FastAPI → client/Chainlit
```

- Tools are dynamically registered at startup via
  `autodiscover_tools` in `app.registry.tools`.
- Parameter extraction and tool planning both leverage the LLM
  through `call_llm` and `extract_parameters_with_llm`.
- `MCPClient.process_message` collects tool outputs, optionally
  formats them with the LLM, then returns the response to the
  FastAPI handler which streams it back to the user.

