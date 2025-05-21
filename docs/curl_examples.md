# cURL Examples

This document lists cURL commands found throughout the repository along with brief explanations of what they do.

## Examples from `app/mcp_server.py`

The following commands appear in the comments of `app/mcp_server.py` as quick tests for the MCP endpoint.

### 1. List available tools
```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"message": "list tools"}'
```
Sends a request to list all registered tools.

### 2. Financial query without parameters
```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"message": "show me top clients"}'
```
Runs a financial query that relies on default parameters.

### 3. Financial query with parameters
```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"message": "show me top clients in USD for Canada region"}'
```
Provides explicit parameters for the financial query.

### 4. More specific query
```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"message": "what are the top gaining clients in CAD currency"}'
```
Queries for top gaining clients with a specified currency.

## Additional commands from the Dockerfile

The repository's Dockerfile also contains a couple of `curl` usages:

- `curl -sL http://deb.nodesource.com/setup_18.x | bash -` – downloads the Node.js setup script during image build.
- `curl -f http://localhost:8000/api/v1/health || exit 1` – health check used by the container to verify the API is running.

