# Docker Usage

This repository provides a `Dockerfile` for running the MCP server and Chainlit UI in a container. The image installs required dependencies and exposes the API and UI ports.

## Build the Image

```bash
docker build -t mcp-server .
```

## Run the Container

```bash
docker run -p 8000:8000 -p 8001:8001 -p 8501:8501 mcp-server
```

When started, the container launches the server via `run.py`. The API will be available on port `8000`, the MCP server on `8001`, and Chainlit on `8501`.

Environment variables such as API keys and cache settings can be provided using a `.env` file or directly passed to `docker run` using `-e` flags.

```
docker run --env-file .env -p 8000:8000 -p 8001:8001 mcp-server
```

Logs are written to `/usr/src/elements-ai-server/logs` inside the container. You can mount this directory to your host to persist them:

```bash
docker run -v $(pwd)/logs:/usr/src/elements-ai-server/logs -p 8000:8000 mcp-server
```
