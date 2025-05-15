# Agent Registration Framework

This document provides a guide for registering external agents with the MCP server.

## Overview

The MCP server supports external agent registration, allowing different agents to provide their own tools, resources, and prompt templates. Each agent operates in its own namespace to avoid conflicts.

## Registration Process

### 1. Register the Agent

First, register your agent with the MCP server to get an agent ID and a namespace:

```python
import requests
import json

# Register a new agent
response = requests.post(
    "http://localhost:8000/api/v1/agents/register",
    json={
        "name": "MyDataAgent",
        "description": "Agent for data processing and analysis",
        "namespace": "data_agent",  # Optional, will use name if not provided
        "capabilities": ["data_processing", "analysis", "visualization"],
        "metadata": {
            "version": "1.0.0",
            "author": "Your Name",
            "contact": "your.email@example.com"
        }
    }
)

# Get the agent ID
agent_data = response.json()
agent_id = agent_data["id"]
namespace = agent_data["namespace"]

print(f"Registered agent with ID: {agent_id}")
print(f"Using namespace: {namespace}")
```

### 2. Register Tools

Your agent can register tools that will be available to the MCP server:

#### Method 1: Using decorators (Python client)

```python
from app.registry.tools import register_tool

@register_tool(
    name="analyze_data",
    description="Analyze data from a CSV file",
    namespace="data_agent",  # Use your agent's namespace
    input_schema={
        "file_path": {"type": "string", "description": "Path to the CSV file"},
        "analysis_type": {"type": "string", "description": "Type of analysis to perform"}
    }
)
async def analyze_data(ctx, file_path: str, analysis_type: str):
    # Tool implementation
    return {
        "result": f"Analysis of {file_path} with {analysis_type}",
        "summary": "Data analysis complete"
    }
```

#### Method 2: Using the API (any client)

```python
# Register a tool via the API
response = requests.post(
    "http://localhost:8000/api/v1/tools/register",
    json={
        "name": "visualize_data",
        "description": "Create a visualization from data",
        "namespace": "data_agent",  # Use your agent's namespace
        "input_schema": {
            "data": {"type": "object", "description": "Data to visualize"},
            "chart_type": {"type": "string", "description": "Type of chart to create"}
        },
        "output_schema": {
            "image_url": {"type": "string", "description": "URL of the generated image"}
        }
    }
)
```

### 3. Register Resources

Your agent can also register external resources:

```python
from app.registry.resources import register_resource
import httpx

# Register a data API resource
data_api = register_resource(
    name="data_api",
    description="API for retrieving data sets",
    handler=httpx.AsyncClient().get,
    namespace="data_agent",  # Use your agent's namespace
    config={
        "base_url": "https://api.example.com/data",
        "api_key": "YOUR_API_KEY",
        "timeout": 10.0
    }
)
```

### 4. Register Prompt Templates

Your agent can provide prompt templates:

```python
from app.registry.prompts import register_prompt

# Register a data analysis prompt template
data_analysis_prompt = register_prompt(
    name="data_analysis",
    description="Prompt for analyzing data patterns",
    template="""
Analyze the following data and identify patterns:

DATA:
{data}

Please provide:
1. Key patterns and trends
2. Anomalies or outliers
3. Recommendations based on the data

ANALYSIS:
""",
    variables=["data"],
    namespace="data_agent"  # Use your agent's namespace
)
```

## Using Registered Components

Once your components are registered, they can be used by the MCP server:

### List Available Components

```python
import requests

# Get list of all registered agents
response = requests.get("http://localhost:8000/api/v1/agents")
agents = response.json()["agents"]

# Get details for a specific agent
agent_id = "your_agent_id"
response = requests.get(f"http://localhost:8000/api/v1/agents/{agent_id}")
agent_details = response.json()

# View components registered by this agent
components = agent_details["components"]
tools = components["tools"]
resources = components["resources"]
prompts = components["prompts"]
```

### Execute a Tool

```python
# Execute a tool from a specific agent
response = requests.post(
    "http://localhost:8000/api/v1/tools/execute",
    json={
        "tool": "data_agent:analyze_data",  # namespace:tool_name
        "parameters": {
            "file_path": "/path/to/data.csv",
            "analysis_type": "trend_analysis"
        }
    }
)
result = response.json()
```

## Complete Agent Integration Example

Here's a complete example of an agent integration:

```python
import requests
import json
import httpx

# Step 1: Register the agent
response = requests.post(
    "http://localhost:8000/api/v1/agents/register",
    json={
        "name": "DataVizAgent",
        "description": "Data visualization and analysis agent",
        "capabilities": ["visualization", "data_processing", "analysis"]
    }
)
agent_data = response.json()
agent_id = agent_data["id"]
namespace = agent_data["namespace"]

print(f"Registered agent with ID: {agent_id}")
print(f"Using namespace: {namespace}")

# Step 2: Register tools via API
tools = [
    {
        "name": "create_chart",
        "description": "Create a chart from data",
        "input_schema": {
            "data": {"type": "array", "description": "Data points"},
            "chart_type": {"type": "string", "description": "Type of chart"},
            "title": {"type": "string", "description": "Chart title"}
        }
    },
    {
        "name": "analyze_trends",
        "description": "Analyze trends in time series data",
        "input_schema": {
            "time_series": {"type": "array", "description": "Time series data"},
            "period": {"type": "string", "description": "Period for analysis"}
        }
    }
]

for tool in tools:
    tool["namespace"] = namespace
    response = requests.post(
        "http://localhost:8000/api/v1/tools/register",
        json=tool
    )
    print(f"Registered tool: {tool['name']}")

# Step 3: Register a prompt template
prompt = {
    "name": "chart_description",
    "description": "Generate a description for a chart",
    "template": "Describe the following chart:\n\nChart type: {chart_type}\nData: {data}\n\nDESCRIPTION:",
    "variables": ["chart_type", "data"],
    "namespace": namespace
}

response = requests.post(
    "http://localhost:8000/api/v1/prompts/register",
    json=prompt
)
print(f"Registered prompt template: {prompt['name']}")

# Now the agent and its components are ready to use
print(f"Agent {agent_data['name']} is ready with namespace {namespace}")
```

## Unregistering an Agent

When your agent is no longer needed, it can be unregistered:

```python
# Unregister an agent
response = requests.delete(f"http://localhost:8000/api/v1/agents/{agent_id}")
print(response.json()["message"])
```

## Best Practices

1. **Use Meaningful Names**: Choose clear, descriptive names for your tools and resources.
2. **Document Parameters**: Provide clear descriptions for all input and output parameters.
3. **Handle Errors**: Ensure your tools handle errors gracefully and return informative error messages.
4. **Respect Namespaces**: Always use your assigned namespace to avoid conflicts.
5. **Version Control**: Include version information in your agent metadata.
6. **Limit Size**: Keep tool responses and parameters reasonably sized to avoid performance issues.
7. **Implement Logging**: Add logging to your tools to help with debugging.
8. **Cleanup Resources**: Unregister your agent when it's no longer needed.

## Technical Details

- Each agent is assigned a unique ID and namespace
- Tools are referenced as `namespace:tool_name`
- Components in the same namespace can access each other
- The MCP server handles routing and execution of tools
- Namespaces prevent conflicts between different agents' components 