# MCP (Model Control Platform) Usage Guide

## Table of Contents
- [Installation](#installation)
- [Methods for Registering Models](#methods-for-registering-models)
  - [Using pip package (Recommended)](#using-pip-package-recommended)
  - [Using Python SDK](#using-python-sdk)
  - [Using REST API directly](#using-rest-api-directly)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)

## Installation

```bash
pip install mcp-client
```

## Methods for Registering Models

### Using pip package (Recommended)

The pip package provides a convenient CLI interface for managing your models:

```bash
# Register a model using configuration file
mcp register model_config.yaml --server-url http://mcp-server:8000 --api-key your-api-key

# List all registered models
mcp list-models

# Get statistics for a specific model
mcp get-stats my-custom-model
```

### Using Python SDK

The Python SDK provides programmatic access with full type safety and async support:

```python
from mcp_client import MCPClient, ModelConfig, ModelBackend
import asyncio

async def register_model():
    # Initialize client
    client = MCPClient(
        base_url="http://mcp-server:8000",
        api_key="your-api-key"
    )
    
    # Create model config
    config = ModelConfig(
        model_id="my-custom-model",
        backend=ModelBackend.CUSTOM,
        api_base="http://my-model:8000",
        additional_params={
            "model_type": "text-generation",
            "team": "nlp-research"
        }
    )
    
    # Register model
    try:
        response = await client.register_model(config)
        print(f"Model registered: {response}")
    finally:
        await client.close()

asyncio.run(register_model())
```

### Using REST API directly

For direct API access without additional dependencies:

```bash
curl -X POST http://mcp-server:8000/models/register \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-custom-model",
    "backend": "custom",
    "api_base": "http://my-model:8000",
    "api_version": "v1"
  }'
```

## Configuration Reference

Example configuration file (`model_config.yaml`):

```yaml
model_id: custom-text-model
backend: custom
api_base: http://model-service:8000
api_version: v1
timeout: 30
max_tokens: 2000
temperature: 0.7
additional_params:
  model_type: text-generation
  description: Custom text generation model
  version: 1.0.0
  team: nlp-research
  environment: production
  resources:
    gpu: true
    memory: 16G
    cpu_cores: 4
  monitoring:
    enable_logging: true
    log_level: info
    metrics_endpoint: /metrics
```

## Key Features

- Easy integration through pip package
- Flexible configuration options
- CLI and SDK availability
- Type safety with Pydantic models
- Async support
- Built-in health checks
- Environment variable support
- Comprehensive monitoring options

## Best Practices

1. Always use version control for your model configurations
2. Store sensitive information in environment variables
3. Implement proper error handling
4. Monitor model performance using the provided metrics
5. Keep configurations in YAML format for better readability
6. Use the async SDK for better performance in production

## Troubleshooting

If you encounter issues:

1. Check your API key and permissions
2. Verify server URL and connectivity
3. Ensure model configuration is valid
4. Check server logs for detailed error messages
5. Use the built-in validation tools before deployment 