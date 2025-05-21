# Graphiti Integration

This folder contains a lightweight integration between the MCP server and the
[Graphiti](https://github.com/getzep/graphiti) project. The goal is to ingest
CRM client data into a local Graphiti instance and query it using natural
language.  A `requirements.txt` file is provided for installing the minimal
dependencies needed to run the example Graphiti backed server.

## Files

- `config.py` – configuration values such as the Graphiti base URL and
  collection ID.
- `client_graphiti.py` – minimal HTTP client used to interact with Graphiti.
- `ingest_clients.py` – script that pulls data from the existing `get_top_clients`
  MCP tool and uploads it to Graphiti.
- `mcp_server_graphiti.py` – an example FastAPI server that exposes a `/mcp`
  endpoint backed by Graphiti.
- `utils.py` – helper functions for parsing tool output.

## Usage

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This installs the packages required for the example MCP server that talks to
   a running Graphiti instance.

2. **Ingest Data**

   ```bash
   python -m graphiti_integration.ingest_clients
   ```

   This calls the `get_top_clients` tool and posts the resulting records to the
   configured Graphiti server.

3. **Run the MCP Server**

   ```bash
   uvicorn mcp_server_graphiti:app --host 0.0.0.0 --port 8080
   ```

   Send a POST request to `/mcp` with a JSON body containing `{"message":
   "Show me all Canadian clients sorted by revenue"}` to query Graphiti.

Configuration values can be overridden using the environment variables
`GRAPHITI_BASE_URL` and `GRAPHITI_COLLECTION_ID`.
