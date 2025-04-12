import typer
import asyncio
import json
from typing import Optional
from .client import MCPClient, ModelConfig, ModelBackend
import yaml

app = typer.Typer()

@app.command()
def register(
    config_file: str = typer.Argument(..., help="Path to model configuration file (YAML/JSON)"),
    server_url: Optional[str] = typer.Option(None, help="MCP server URL"),
    api_key: Optional[str] = typer.Option(None, help="API key for authentication")
):
    """Register a model with MCP using a configuration file."""
    # Load configuration
    with open(config_file) as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    # Create model config
    config = ModelConfig(**config_data)
    
    # Register model
    async def _register():
        client = MCPClient(base_url=server_url, api_key=api_key)
        try:
            response = await client.register_model(config)
            typer.echo(f"Successfully registered model {config.model_id}")
            typer.echo(f"Response: {json.dumps(response, indent=2)}")
        finally:
            await client.close()
    
    asyncio.run(_register())

@app.command()
def list_models(
    server_url: Optional[str] = typer.Option(None, help="MCP server URL"),
    api_key: Optional[str] = typer.Option(None, help="API key for authentication")
):
    """List all registered models."""
    async def _list():
        client = MCPClient(base_url=server_url, api_key=api_key)
        try:
            models = await client.list_models()
            typer.echo(json.dumps(models, indent=2))
        finally:
            await client.close()
    
    asyncio.run(_list())

@app.command()
def get_stats(
    model_id: str = typer.Argument(..., help="Model ID to get statistics for"),
    server_url: Optional[str] = typer.Option(None, help="MCP server URL"),
    api_key: Optional[str] = typer.Option(None, help="API key for authentication")
):
    """Get usage statistics for a model."""
    async def _stats():
        client = MCPClient(base_url=server_url, api_key=api_key)
        try:
            stats = await client.get_model_stats(model_id)
            typer.echo(json.dumps(stats, indent=2))
        finally:
            await client.close()
    
    asyncio.run(_stats())

if __name__ == "__main__":
    app() 