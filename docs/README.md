# MCP Documentation

This directory contains supplementary guides for using and extending the MCP server.

## Agent and Tool Registration

Agents can register themselves with the FastAPI service and then register their tools and prompts. The full process and code samples are provided in [AGENT_REGISTRATION.md](../AGENT_REGISTRATION.md).

At a high level you:
1. **Register the agent** at `/api/v1/agents/register` to obtain an ID and namespace.
2. **Register tools** under that namespace using `/api/v1/tools/register`.
3. **Optionally register resources and prompt templates**.

Once registered, tools can be executed by referencing `namespace:tool_name` in requests.

## Indexing Tools with Cohere Compass

To make tools discoverable by Compass, run [`app/utils/setup_compass_index.py`](../app/utils/setup_compass_index.py). This script collects all registered tools and resources, derives categories and intents, and uploads them to your Compass index.

Execute it with:
```bash
python app/utils/setup_compass_index.py
```
Ensure your Compass credentials (`COMPASS_API_URL`, `COMPASS_BEARER_TOKEN`, `COMPASS_INDEX_NAME`) are configured in the environment or `.env` file.

## Chained Tool Execution

The MCP server can chain multiple tools in one request. Example prompt:
```
Get the top clients for last quarter and analyze their sentiment.
```
This could call a financial data tool followed by a sentiment analysis tool.

Another example:
```
Find recent support tickets from our biggest customers and summarize the key complaints.
```
The planner will automatically create a sequence of tool calls to fulfill the request.

## Current Limitations

- Tool chaining is sequential only; parallel execution is not yet supported.
- Error handling for multi-step plans is minimal—if one tool fails, the chain stops.
- Compass indexing must be triggered manually after new tools are added.

## Planned Improvements

- Support for parallel tool execution and better rollback on failure.
- Automatic indexing of newly registered tools in Compass.
- Richer plan validation to catch parameter issues earlier.

## Further Reading

See [curl_examples.md](curl_examples.md) for practical API calls using `curl`.
