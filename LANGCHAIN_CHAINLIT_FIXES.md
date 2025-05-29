# LangChain + Chainlit Integration Fixes  
*(MCP-Server 0.1.0 – May 2025)*  

## 1  Problem Statement  
| Area | Symptoms |
|------|----------|
| LangChain routing | • Agent rarely selected any MCP tools<br>• Parameters were not passed, causing `TypeError`<br>• Multi-step plans were dropped |
| Chainlit UI | • “No tools available” message even when server had them<br>• Tool calls used deprecated `mcp_session.call_tool` – broke after SSE switch<br>• Markdown tables rendered as plain text |

*… sections 2-8 unchanged …*

## 9  Deployment Commands  

After installing the package **globally** (or inside a container/venv):

| Mode | Python CLI (uvx) | Node CLI (npx) | Notes |
|------|------------------|----------------|-------|
| HTTP API (+ SSE) | `uvx mcp-server --host 0.0.0.0 --port 8080` | `npx mcp-server --host 0.0.0.0 --port 8080` | Starts FastAPI on :8080. Add `--no-ui` to skip Chainlit. |
| STDIO (agent embedding) | `uvx mcp-server --mode stdio` | `npx mcp-server --mode stdio` | Useful when parent process handles transport (e.g., Compass, Graphiti). |
| Mock data | `uvx mock --port 8001` | `npx mcp mock --port 8001` | Spins up local financial endpoints. |

### Container example
```bash
docker run -p 8080:8080 your-image \
  uvx mcp-server --host 0.0.0.0 --port 8080
```

### Systemd unit snippet
```
ExecStart=/usr/local/bin/uvx mcp-server --host 0.0.0.0 --port 8080
Restart=always
```

## 10  Quick-Start (local dev)

```bash
# 1. install
pip install -e .          # or: npm install -g mcp-server
uvx install               # sets up extras

# 2. run everything
uvx mcp-server --host localhost --port 8080 &   # API
npx mcp run           # Chainlit UI (or uvx run if preferred)
```

## 11  Migration Notes  
1. Replace old `uvx run`/`npm start` with the **full package** invocation shown above.  
2. Make sure `pandas`, `langchain>=0.1.15` are inside your runtime image.  
3. ENV vars consumed: `LLM_MODEL`, `MCP_SERVER_PORT`, etc.

*Other sections unchanged.*