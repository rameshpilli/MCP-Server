"""
CLI module for the MCP Server.
Provides the `uvx` command for running the Chainlit app with a generic MCP server.
"""

import os
import sys
import click
import subprocess
import time
from pathlib import Path
from typing import Optional

def print_server_urls(host: str, port: int, style: str = "info"):
    """Print server URLs with consistent styling"""
    api_url = f"http://{host}:{port}"
    ui_url = f"http://{host}:{port + 1}"
    
    if style == "info":
        click.echo("\n" + "=" * 60)
        click.echo("📡 Server URLs:")
        click.echo(f"🔗 API Server: {click.style(api_url, fg='blue', underline=True)}")
        click.echo(f"🔗 Chainlit UI: {click.style(ui_url, fg='blue', underline=True)}")
        click.echo("=" * 60 + "\n")
    elif style == "ready":
        click.echo("\n" + "-" * 60)
        click.echo("✨ Servers are ready!")
        click.echo(f"👉 API Server: {click.style(api_url, fg='green', bold=True)}")
        click.echo(f"👉 Chainlit UI: {click.style(ui_url, fg='green', bold=True)}")
        click.echo("Press Ctrl+C to stop the servers")
        click.echo("-" * 60 + "\n")

@click.group()
def cli():
    """MCP Server CLI"""
    pass

@cli.command(name='run')
@click.option('--host', default='localhost', help='Host to run the server on')
@click.option('--port', default=8000, help='Port to run the server on')
@click.option('--env', default='.env', help='Path to environment file')
def run_server(host: str, port: int, env: str):
    """Run the MCP Server with Chainlit UI"""
    try:
        # Ensure we're in the project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Load environment variables
        if os.path.exists(env):
            from dotenv import load_dotenv
            load_dotenv(env)
            click.echo(click.style(f"✓ Loaded environment from {env}", fg='green'))

        # Print initial server URLs
        print_server_urls(host, port)

        # Start the FastAPI server in the background
        server_cmd = [
            "uvicorn",
            "app.main:app",
            "--host", host,
            "--port", str(port),
            "--reload"
        ]
        
        click.echo(click.style("Starting FastAPI server...", fg='yellow'))
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Start Chainlit
        chainlit_cmd = [
            "chainlit",
            "run",
            "ui/app.py",
            "--host", host,
            "--port", str(port + 1)
        ]
        
        click.echo(click.style("Starting Chainlit UI...", fg='yellow'))
        chainlit_process = subprocess.Popen(
            chainlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Monitor both processes
        try:
            # Wait for servers to start
            time.sleep(2)
            
            # Track server readiness
            api_ready = False
            chainlit_ready = False
            
            while True:
                # Check server status
                if server_process.poll() is not None:
                    click.echo(click.style("❌ FastAPI server stopped unexpectedly", fg='red'))
                    break

                if chainlit_process.poll() is not None:
                    click.echo(click.style("❌ Chainlit stopped unexpectedly", fg='red'))
                    break

                # Process FastAPI output
                api_output = server_process.stdout.readline()
                if api_output:
                    if "Application startup complete" in api_output and not api_ready:
                        api_ready = True
                        click.echo(click.style("✓ FastAPI server is ready!", fg='green'))
                    click.echo(f"FastAPI: {api_output.strip()}")

                # Process Chainlit output
                chainlit_output = chainlit_process.stdout.readline()
                if chainlit_output:
                    if "Chainlit app running" in chainlit_output and not chainlit_ready:
                        chainlit_ready = True
                        click.echo(click.style("✓ Chainlit UI is ready!", fg='green'))
                    click.echo(f"Chainlit: {chainlit_output.strip()}")

                # If both servers are ready, print URLs one final time
                if api_ready and chainlit_ready and not hasattr(run_server, 'urls_printed'):
                    print_server_urls(host, port, style="ready")
                    run_server.urls_printed = True

        except KeyboardInterrupt:
            click.echo("\n" + click.style("Shutting down servers...", fg='yellow'))
            server_process.terminate()
            chainlit_process.terminate()
            server_process.wait()
            chainlit_process.wait()
            click.echo(click.style("✓ Servers stopped", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='init')
def init_project():
    """Initialize the MCP Server project"""
    try:
        # Create necessary directories
        dirs = [
            "app/tools",
            "app/utils",
            "app/registry",
            "tests",
            "project-docs",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            click.echo(click.style(f"✓ Created directory: {dir_path}", fg='green'))

        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists():
            env_file.write_text("""# MCP Server Environment Variables
SERVER_NAME="MCP Server"
SERVER_DESCRIPTION="Generic MCP Server with Chainlit Integration"
API_PREFIX="/api/v1"
CORS_ORIGINS=["http://localhost:8000", "http://localhost:8001"]

# Dummy API Keys (for testing only)
OPENAI_API_KEY="dummy-key-123"
COHERE_API_KEY="dummy-key-456"
""")
            click.echo(click.style("✓ Created .env file", fg='green'))

        # Create a dummy tools file for testing
        tools_file = Path("app/tools/dummy_tools.py")
        if not tools_file.exists():
            tools_file.write_text('''
"""
Dummy tools for testing the MCP server.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from fastmcp import Tool, ToolRegistry

class DummyCategoryEnum(str, Enum):
    """Categories for dummy data"""
    TEST = "test"
    DEMO = "demo"
    SAMPLE = "sample"

class DummyData(BaseModel):
    """Model for dummy data"""
    id: str
    name: str
    category: DummyCategoryEnum
    value: float
    tags: List[str]

@ToolRegistry.register()
class GetDummyData(Tool):
    """Get dummy data for testing"""
    name = "get_dummy_data"
    description = "Retrieve dummy data for testing purposes"
    namespace = "dummy"
    
    class Input(BaseModel):
        category: Optional[DummyCategoryEnum] = None
        limit: int = 10
    
    class Output(BaseModel):
        data: List[DummyData]
    
    def execute(self, input: Input) -> Output:
        # Generate dummy data
        dummy_data = [
            DummyData(
                id=f"item_{i}",
                name=f"Test Item {i}",
                category=DummyCategoryEnum.TEST,
                value=float(i) * 1.5,
                tags=["test", "dummy"]
            )
            for i in range(input.limit)
        ]
        return self.Output(data=dummy_data)

@ToolRegistry.register()
class EchoTool(Tool):
    """Echo tool for testing"""
    name = "echo"
    description = "Echo back the input message"
    namespace = "dummy"
    
    class Input(BaseModel):
        message: str
    
    class Output(BaseModel):
        echo: str
    
    def execute(self, input: Input) -> Output:
        return self.Output(echo=f"Echo: {input.message}")
''')
            click.echo(click.style("✓ Created dummy tools file", fg='green'))

        click.echo(click.style("✓ Project initialized successfully!", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error initializing project: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='mcp')
def run_mcp():
    """Run the MCP server in stdio mode for Chainlit integration"""
    try:
        # Ensure we're in the project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        # Set environment variables for stdio mode
        os.environ["TRANSPORT"] = "stdio"
        
        click.echo(click.style("Starting MCP server in stdio mode...", fg='yellow'))
        
        # Import and run the MCP server
        from app.mcp_server import main as mcp_main
        import asyncio
        
        # Run the MCP server
        asyncio.run(mcp_main())
    except Exception as e:
        click.echo(click.style(f"Error running MCP server: {str(e)}", fg='red'), err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli() 