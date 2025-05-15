# Sample MCP Agent Template

This is a template project for creating an agent that integrates with the MCP server.

## Project Structure

```
agent_template/
├── README.md             # This file
├── agent.py              # Main agent registration
├── requirements.txt      # Dependencies
├── config.py             # Configuration
├── tools/                # Tool definitions
│   ├── __init__.py
│   ├── search.py         # Search-related tools
│   └── processing.py     # Data processing tools
├── resources/            # Resource definitions
│   ├── __init__.py
│   └── apis.py           # API resource definitions
└── prompts/              # Prompt templates
    ├── __init__.py
    └── templates.py      # Prompt templates
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update the configuration in `config.py` with your MCP server details.

3. Run the agent registration:
```bash
python agent.py
```

## Customizing Your Agent

1. Edit `agent.py` to update your agent's name, description, and capabilities.
2. Add your own tools in the `tools/` directory.
3. Define resources in the `resources/` directory.
4. Create prompt templates in the `prompts/` directory.

## Example Usage

Once your agent is registered, you can use its tools through the MCP server:

```python
import requests

# Execute a tool from your agent
response = requests.post(
    "http://localhost:8000/api/v1/tools/execute",
    json={
        "tool": "your_namespace:your_tool",
        "parameters": {
            "param1": "value1",
            "param2": "value2"
        }
    }
)
result = response.json()
print(result)
```

## Adding New Tools

To add a new tool:

1. Create a function in one of the tool modules or add a new module in the `tools/` directory.
2. Register the tool using the `@register_tool` decorator.
3. Import the tool module in `tools/__init__.py`.

Example:

```python
from app.registry.tools import register_tool

@register_tool(
    name="my_new_tool",
    description="Description of what my tool does",
    namespace="your_agent_namespace",
    input_schema={
        "param1": {"type": "string", "description": "Description of param1"},
        "param2": {"type": "integer", "description": "Description of param2"}
    }
)
async def my_new_tool(ctx, param1: str, param2: int):
    # Tool implementation
    return {
        "result": f"Processed {param1} with {param2}",
    }
```

## Updating Your Agent

To update your agent's metadata or capabilities:

```python
import requests

response = requests.patch(
    f"http://localhost:8000/api/v1/agents/{agent_id}",
    json={
        "description": "Updated description",
        "capabilities": ["updated", "capability", "list"]
    }
)
``` 