# MCP Server Integration Guide

## Table of Contents
1. [Storage Backend Configuration](#storage-backend-configuration)
2. [LLM Integration](#llm-integration)
3. [API Reference](#api-reference)
4. [Examples](#examples)

## Storage Backend Configuration

### Local Storage
```python
from app.core.storage import LocalStorageBackend

storage = LocalStorageBackend(base_path="/path/to/storage")
```

### S3-Compatible Storage (MinIO, Ceph, etc.)
```python
from app.core.storage import S3StorageBackend

# For AWS S3
storage = S3StorageBackend(
    bucket="my-bucket",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    region_name="us-east-1"
)

# For MinIO or other S3-compatible storage
storage = S3StorageBackend(
    bucket="my-bucket",
    endpoint_url="http://minio.local:9000",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key"
)
```

### Azure Data Lake Storage
```python
from app.core.storage import AzureStorageBackend

storage = AzureStorageBackend(
    connection_string="your-connection-string",
    container="your-container"
)
```

## LLM Integration

### Using with OpenAI
```python
import openai
from mcp_client import MCPClient

class LLMAssistant:
    def __init__(self):
        self.mcp = MCPClient()
        openai.api_key = "your-api-key"
    
    async def process_query(self, query: str):
        # Get context from MCP server
        context = await self.get_context(query)
        
        # Create OpenAI prompt
        prompt = f"""Context: {context}
        
        Query: {query}
        
        Please help with this query using the provided context."""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    async def get_context(self, query: str):
        # Example: Get relevant files based on query
        response = await self.mcp.execute_task(
            "list_directory",
            {"path": "docs"}
        )
        
        # Get file contents for relevant files
        context = []
        for file in response["data"]["contents"]:
            if file["is_file"]:
                content = await self.mcp.execute_task(
                    "read_file",
                    {"file_path": file["path"]}
                )
                context.append(content["data"]["content"])
        
        return "\n".join(context)
```

### Using with Langchain
```python
from langchain.agents import Tool
from langchain.agents import AgentExecutor, AgentType
from langchain.llms import OpenAI
from mcp_client import MCPClient

def create_mcp_agent():
    mcp = MCPClient()
    
    tools = [
        Tool(
            name="ListFiles",
            func=lambda p: mcp.execute_task("list_directory", {"path": p}),
            description="List files in a directory"
        ),
        Tool(
            name="ReadFile",
            func=lambda p: mcp.execute_task("read_file", {"file_path": p}),
            description="Read contents of a file"
        ),
        Tool(
            name="SearchFiles",
            func=lambda q: mcp.execute_task("search_files", {"query": q}),
            description="Search for files matching a pattern"
        )
    ]
    
    llm = OpenAI(temperature=0)
    agent = AgentExecutor.from_agent_and_tools(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm
    )
    
    return agent
```

## API Reference

### Task Types

1. File Operations:
```json
{
    "task_type": "list_directory",
    "parameters": {
        "path": "path/to/dir"
    }
}

{
    "task_type": "read_file",
    "parameters": {
        "file_path": "path/to/file"
    }
}

{
    "task_type": "write_file",
    "parameters": {
        "file_path": "path/to/file",
        "content": "base64_encoded_content"
    }
}
```

2. External APIs:
```json
{
    "task_type": "get_joke",
    "parameters": {}
}

{
    "task_type": "get_crypto_prices",
    "parameters": {}
}
```

### Response Format
```json
{
    "status": "success",
    "timestamp": "2024-04-12T18:04:06.279Z",
    "cached": false,
    "data": {
        // Task-specific response data
    }
}
```

## Examples

### Python Example
```python
from mcp_client import MCPClient

async def main():
    client = MCPClient()
    
    # List files
    files = await client.execute_task(
        "list_directory",
        {"path": "docs"}
    )
    
    # Read file
    content = await client.execute_task(
        "read_file",
        {"file_path": "docs/readme.md"}
    )
    
    # Use external API
    joke = await client.execute_task(
        "get_joke",
        {}
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### curl Example
```bash
# List directory
curl -X POST http://localhost:8000/execute-task \
    -H "Content-Type: application/json" \
    -H "X-API-Key: test_key" \
    -d '{"task_type": "list_directory", "parameters": {"path": "docs"}}'

# Read file
curl -X POST http://localhost:8000/execute-task \
    -H "Content-Type: application/json" \
    -H "X-API-Key: test_key" \
    -d '{"task_type": "read_file", "parameters": {"file_path": "docs/readme.md"}}'
```

### JavaScript Example
```javascript
async function mcpRequest(taskType, parameters = {}) {
    const response = await fetch('http://localhost:8000/execute-task', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test_key'
        },
        body: JSON.stringify({
            task_type: taskType,
            parameters: parameters
        })
    });
    
    return await response.json();
}

// Example usage
async function main() {
    const files = await mcpRequest('list_directory', { path: 'docs' });
    console.log('Files:', files);
    
    const content = await mcpRequest('read_file', { file_path: 'docs/readme.md' });
    console.log('Content:', content);
} 