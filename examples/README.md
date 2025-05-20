# Examples

This directory contains helper scripts for running the MCP server locally.

## Dummy Financial Server

`dummy_financial_server.py` provides mock endpoints for the ClientView financial
tools. It hosts three in-memory tables with sample data so the tools can be
tested without access to corporate services.

### Running the Server

```bash
uvicorn examples.dummy_financial_server:app --host 0.0.0.0 --port 8001
```

Set the `CLIENTVIEW_BASE_URL` environment variable so the tools point to this
server:

```bash
export CLIENTVIEW_BASE_URL="http://localhost:8001"
```

After starting the server and setting the environment variable, calls to the
financial tools will return the dummy data.
