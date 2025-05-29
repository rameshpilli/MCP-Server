# MCP Server 🚀  
A streamlined **Model Context Protocol (MCP)** server that plugs a Large-Language-Model into real-world data through _tools_, _agents_ and _rich streaming APIs_.  
It comes with:

* 🔧 Automatic tool discovery & chaining (FastMCP, LangChain or Compass routing)  
* 📊 First-class table output (Markdown, HTML, CSV, JSON) for beautiful UIs  
* 🌐 FastAPI backend with Swagger/OpenAPI + SSE streaming  
* 🖥️ Chainlit front-end out of the box  
* 🛠️ `uvx` / `npm` CLIs for local dev, mocks & deployment  
* 🐳 Docker & Kubernetes manifests for prod

---

## 1. Quick-Start

```bash
# One-liner: install + dev-run
python install.py --dev --venv .venv   # creates venv, installs everything
source .venv/bin/activate
uvx mcp-server --host localhost --port 8080 &   # API
npx mcp run                                     # Chainlit UI
```

---

## 2. Installation   <!-- updated -->

### 2.1 Recommended – single script

```bash
# basic (global) install
python install.py

# create / use virtualenv
python install.py --venv .venv --dev
```

The script:

1.  Checks Python ≥ 3.9, Node & npm
2.  Creates a virtual-env if `--venv` is supplied
3.  Installs the Python package **crm-mcp-server**
4.  Installs extra requirements (pandas, langchain …)
5.  Installs or updates the Node CLI wrappers  
    (`mcp`, `mcp-server`, `mcp-tools`)
6.  Prints the exact commands to start the server in HTTP or STDIO mode

Run `python install.py --help` for all options (`--method pip|uvx|npm`, `--upgrade`, `--no-deps`, …).

### 2.2 Manual alternatives  

| Method | Command |
|--------|---------|
| pip    | `pip install crm-mcp-server` |
| uvx    | `uvx install crm-mcp-server` |
| npm    | `npm install -g mcp-server`  |

> After a manual install you still start the server the same way:  
> `uvx mcp-server …` **or** `npx mcp-server …`

---

## 3. Running the Server

| Mode | Python CLI (uvx) | Node CLI (npx) | Notes |
|------|------------------|----------------|-------|
| HTTP API (+ SSE) | `uvx mcp-server --host 0.0.0.0 --port 8080` | `npx mcp-server --host 0.0.0.0 --port 8080` | Start FastAPI. Add `--no-ui` to skip Chainlit |
| STDIO (agent embedding) | `uvx mcp-server --mode stdio` | `npx mcp-server --mode stdio` | When another parent process handles transport |
| Mock data server | `uvx mock --port 8001` | `npx mcp mock --port 8001` | Local financial endpoints |

---

## 4. CLI Usage (cheat-sheet)

```bash
uvx mcp-server --help      # all runtime flags
uvx mock                   # start mock endpoints
mcp-tools list             # inspect tools
mcp-tools execute getTopClients --params '{"region":"USA"}'
```

---

## 5. Tool Discovery & Chaining

Drop any Python file in `app/tools/` and decorate with `@mcp.tool()` **or** expose
`register_tools(mcp)`.  
Tools from `app/tools/clientview_financials.py` are auto-registered and usable from the UI.

---

## 6. API End-Points

Swagger: `http://localhost:8080/docs`  

| Path | Method | Purpose |
|------|--------|---------|
| `/mcp` | POST | main NL → tools |
| `/execute` | POST | direct tool call |
| `/tools` | GET | tool catalogue |
| `/stream` | POST | SSE streaming |
| `/ping` | GET | health |

---

## 7. Deployment

```bash
docker build -t mcp-server .
docker run -p 8080:8080 mcp-server \
  uvx mcp-server --host 0.0.0.0 --port 8080
```

or systemd:

```
ExecStart=/usr/local/bin/uvx mcp-server --host 0.0.0.0 --port 8080
Restart=always
```

---

### More docs
*   `LANGCHAIN_CHAINLIT_FIXES.md` – deep-dive into the LangChain + Chainlit patch
*   `docs/` – architecture, API reference, Docker/K8s guides

© 2025 MCP Team – MIT licence  
