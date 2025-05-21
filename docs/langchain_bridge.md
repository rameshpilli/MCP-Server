# LangChainBridge

The `LangChainBridge` module introduces an experimental planning layer based on
[LangChain](https://python.langchain.com/).  It exposes the same public methods
as `MCPBridge` so it can be used as a drop-in replacement while evaluating
LangChain agents.

## Usage

```python
from app.langchain_bridge import LangChainBridge
bridge = LangChainBridge()
```

`LangChainBridge` converts registered MCP tools into LangChain `Tool` objects and
runs a zero‑shot agent to build a plan.  If planning fails the class falls back
to the existing `MCPBridge` routing logic.

## Sample Comparison

The script [`examples/langchain_comparison.py`](../examples/langchain_comparison.py)
prints planning output for a few queries using both bridges.  Below is an example
when the environment has a working LLM configuration:

```
Query: Top clients in Canada
MCPBridge plan: [{'tool': 'clientview:get_top_clients', 'parameters': {'region': 'CA'}}]
LangChain plan: [{'tool': 'clientview:get_top_clients', 'parameters': {'region': 'CA'}}]
```

Actual plans depend on the underlying language model and available tools but
should resemble each other closely.

## Advantages

- Utilises LangChain’s ecosystem of agents and planning utilities.
- Easier experimentation with different agent types (e.g. ReAct).
- Reuses existing tool registry without modification.

## Limitations

- Adds an additional dependency and slightly higher overhead.
- LangChain planning APIs are still evolving and may change.
- Requires network access for the language model, just like the existing bridge.
