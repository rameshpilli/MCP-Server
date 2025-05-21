# Model Context Protocol (MCP) Server with Agent Registration

This project implements a Model Context Protocol (MCP) server with a Chainlit UI and FastAPI backend that supports agent registration. It allows various tools, resources, and prompt templates to be registered and used by agents.

## Architecture

The system follows a layered architecture:

```
┌────────────────┐
│  Chainlit UI   │
└────────────────┘
        │
        ▼
┌────────────────┐
│   MCP Client   │
└────────────────┘
        │
        ▼
┌────────────────┐
│ FastAPI Server │
└────────────────┘
        │
        ▼
┌────────────────┐
│   MCP Server   │
└────────────────┘
        │
┌───────────────────────────────┐
│                               │
▼                               ▼
┌─────────────┐             ┌──────────────┐
│  External   │             │   Internal   │
│   Agents    │             │ Components   │
└─────────────┘             └──────────────┘
      │                             │
      ▼                             ▼
┌──────────────────┬───────────────────┬──────────────────┐
│      Tools       │     Resources     │     Prompts      │
└──────────────────┴───────────────────┴──────────────────┘
```

### Components

1. **Chainlit UI (ui/app.py)**
   - Web interface for user interaction
   - Sends messages to the MCP client

2. **MCP Client (ui/mcp/client.py)**
   - Processes user messages
   - Sends requests to the MCP server
   - Handles context retrieval from Cohere

3. **FastAPI Server (app/main.py)**
   - API endpoints for agent registration
   - Forwards requests to MCP Server
   - Manages agent metadata

4. **MCP Server (app/mcp_server.py)**
   - Central orchestration engine
   - Executes tools and processes requests
   - Uses namespaced registries for tools, resources, and prompts

5. **Registry (app/registry/)**
   - Management of tools, resources, and prompts
   - Namespace support for multi-agent environments
   - Dynamic registration and discovery

### Flow

1. User inputs a message in Chainlit UI
2. Message is sent to MCP Client
3. MCP Client forwards request to FastAPI Server
4. FastAPI Server forwards to MCP Server
5. MCP Server processes request, executes tools as needed
6. Response is returned through the chain
7. Response is displayed in Chainlit UI

## Agent Registration

The system supports registration of external agents, each with their own tools,  resources, and prompts.

### How to Register an Agent

```python
import requests

# Register a new agent
response = requests.post(
    "http://localhost:8000/api/v1/agents/register",
    json={
        "name": "MyAgent",
        "description": "My custom agent for data processing",
        "namespace": "myagent",
        "capabilities": ["search", "summarize", "analyze"]
    }
)
agent_id = response.json()["id"]
```

### Namespaced Components

All components (tools, resources, prompts) are namespaced to avoid conflicts:

```python
# Register a tool with namespace
from app.registry.tools import register_tool

@register_tool(
    name="custom_search",
    description="Custom search implementation",
    namespace="myagent"
)
async def custom_search(query: str):
    # Search implementation
    pass
```

## Setup

1. Install dependencies from `pyproject.toml`:
```bash
pip install .
```
This project uses `pyproject.toml` as the single source of truth for
dependencies. The provided Dockerfile and Kubernetes manifests install
packages the same way using `pip install .`.

2. Configure environment variables (create a `.env` file). You can start by copying
   `.env.example` and then filling in the required values (e.g. your OpenAI API key):
```
# MCP Server
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8080

# FastAPI Server
HOST=localhost
PORT=8000

# LLM Configuration
LLM_MODEL=claude-3-opus-20240229
LLM_BASE_URL=https://api.anthropic.com/v1/messages
# Add OAuth settings if needed

# Cohere Configuration (optional)
COHERE_INDEX_NAME=mcp_index
COHERE_SERVER_URL=
COHERE_SERVER_BEARER_TOKEN=
```

## Running the Application

### Option 1: Start Both Servers with Single Command

```bash
python run.py
```

### Option 2: Start Each Server Separately

1. Start the MCP server:
```bash
python app/mcp_server.py
```

2. Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Start the Chainlit UI:
```bash
chainlit run ui/app.py
```

### Running the Dummy Financial Server

To test the financial tools without corporate services, start the mock API:

```bash
uvicorn examples.dummy_financial_server:app --host 0.0.0.0 --port 8001
```

Set `CLIENTVIEW_BASE_URL` to point the tools to this server:

```bash
export CLIENTVIEW_BASE_URL="http://localhost:8001"
```

4. Access the UI at http://localhost:8501
5. Access the API docs at http://localhost:8000/docs

### Docker Usage

Build the container image:
```bash
docker build -t mcp-server .
```

The Dockerfile installs all dependencies using `pip install .`, so the
`pyproject.toml` is the single source of package versions.

Run the server using your `.env` file:
```bash
docker run --env-file .env -p 8000:8000 -p 8081:8081 -p 8501:8501 mcp-server
```

## Adding Documents

Place documents in the `docs/` directory. The system supports Markdown (`.md`) and text (`.txt`) files.

## API Endpoints

- **GET /api/v1/health** - Health check
- **POST /api/v1/chat** - Chat with the MCP server
- **POST /api/v1/agents/register** - Register a new agent
- **GET /api/v1/agents** - List all registered agents
- **GET /api/v1/agents/{agent_id}** - Get agent information
- **DELETE /api/v1/agents/{agent_id}** - Unregister an agent

## Configuration

Edit `app/config.py` to change configuration settings.

### Logging

Logging behavior is controlled by two environment variables:

- `LOG_LEVEL` sets the verbosity (default `INFO`).
- `LOG_TO_STDOUT_ONLY` if set, disables file logging and writes logs only to stdout.

## Project Structure

```
mcp-app/
├── app/
│   ├── main.py                # FastAPI entrypoint that handles HTTP requests and routes them to the right components.
│   ├── mcp_client.py          # The brain of the operation - processes prompts and coordinates tool execution.
│   ├── mcp_bridge.py          # Connects to Cohere Compass to understand user intent and plan tool usage.
│   ├── registry/
│   │   ├── tools.py           # A catalog of available tools that can be used to help users.
│   │   ├── prompts.py         # Templates for consistent communication with users and tools.
│   │   └── resources.py       # External service connections like CRM or databases.
│   ├── memory/
│   │   └── short_term.py      # Keeps track of conversation context using Redis.
│   ├── workers/
│   │   └── summarizer.py      # Example tool that processes and summarizes information.
│   └── chaining.py            # Orchestrates multiple tools working together.
├── ui/
│   └── app.py                 # A friendly chat interface for users to interact with the system.
├── .env                       # Configuration secrets and API keys (keep this safe!).
├── Dockerfile                 # Instructions for packaging the app into a container.
├── pyproject.toml             # Project metadata and dependencies.
└── kubernetes/
    ├── deployment.yaml        # Tells Kubernetes how to run multiple copies of the app.
    ├── service.yaml           # Sets up networking so other services can talk to the app.
    └── ingress.yaml           # Manages external access to the app.
```

## Development Phases

1. ✅ UI Setup (Chainlit)
2. ⏳ FastAPI Server Setup
3. ⏳ MCP Client Logic
4. ⏳ Cohere Compass Integration
5. ⏳ Tool Chaining Logic
6. ⏳ Sample Tools Implementation
7. ⏳ Session Memory
8. ⏳ Testing Setup
9. ⏳ Docker Configuration - see [docs/docker_usage.md](docs/docker_usage.md)
10. ⏳ Kubernetes Deployment

## API Endpoints

- POST `/chat` - Main chat endpoint
- POST `/register` - Register new tools
- POST `/tool/{tool_name}` - Execute specific tool 



 CRM MCP Server Request Flow

1. Client Request
	•	The user (Chainlit UI or API consumer) sends a request (e.g., a message or command).

2. main.py (FastAPI)
	•	Receives the HTTP request at /api/v1/chat
	•	Calls mcp_client.process_message()

3. mcp_bridge.py
	•	Acts as a bridge between FastAPI and the MCP server.
	•	Uses Cohere Compass to classify the intent and decide which tools to run.
	•	Returns a routing plan (tools + parameters).

4. mcp_server.py (FastMCP Server)
	•	Receives the tool execution plan from the bridge.
	•	Finds the matching tools from its registry.
	•	Executes the tools (can support chaining).

5. Tools
	•	Tools do the real work (e.g., fetch financial data, read docs).
	•	Return results to the MCP server.

6. Backflow
	•	Tools → MCP Server: Results go back to the FastMCP server.
	•	MCP Server → mcp_bridge.py: FastMCP hands over the tool results.
	•	mcp_bridge.py → main.py: Bridge optionally uses generate_response() to format the result.
	•	main.py → Client: Final response is returned to the UI/API caller.

⸻

🧠 Key Insight

The MCP Server is the brain, the Bridge is the router & formatter, and FastAPI is just the door.
## Python SDK Usage

You can interact with the server from Python using the `MCPClient` class:
```python
from mcp_client import MCPClient

client = MCPClient(base_url="http://mcp-server:8000")
response = client.query_sync("Top clients in Canada")
print(response)
```

Use the `MCP_SERVER_URL` environment variable to configure the default server URL. When unset, it falls back to `http://localhost:8000`.

For a more in-depth explanation of the codebase see
[docs/code_walkthrough.md](docs/code_walkthrough.md).
