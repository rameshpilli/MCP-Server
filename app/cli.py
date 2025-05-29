"""uvx command line interface.

This module exposes the ``uvx`` command which is used to manage and run the CRM
MCP server. The command group provides the following subcommands:

* ``uvx run`` – start the FastAPI backend and the Chainlit UI together. This is
  the main entry point when developing or testing the application.
* ``uvx init`` – create the folders and configuration files required for a new
  project. It sets up the environment so that the server can run.
* ``uvx install`` - install the MCP server and all dependencies.
* ``uvx update`` - update the MCP server to the latest version.
* ``uvx status`` - check the status of running MCP server instances.
* ``uvx mock`` - run mock financial endpoints for testing.
* ``uvx format-table`` - format table data from files or stdin.
* ``uvx crm-mcp`` – run the server in ``stdio`` mode to integrate with
  Chainlit's local tooling.

Use ``uvx --help`` to see all available options for each subcommand.
"""

import os
import sys
import click
import subprocess
import time
import json
import logging
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Set up logging
logger = logging.getLogger("mcp_cli")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants
DEFAULT_HOST = "localhost"
DEFAULT_API_PORT = 8000
DEFAULT_UI_PORT = 8001
DEFAULT_MOCK_PORT = 8001
DEFAULT_TABLE_FORMAT = "markdown"
VALID_TABLE_FORMATS = ["markdown", "html", "csv", "json", "table"]

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

def get_project_root() -> Path:
    """Get the project root directory"""
    # First check if we're in the project root
    current_dir = Path.cwd()
    if (current_dir / "app").exists() and (current_dir / "pyproject.toml").exists():
        return current_dir
    
    # Check if we're in the app directory
    if current_dir.name == "app" and (current_dir.parent / "pyproject.toml").exists():
        return current_dir.parent
    
    # Try to find project root by looking for pyproject.toml
    try:
        file_path = Path(__file__)
        if file_path.exists():
            # If we're running as a module
            return file_path.parent.parent
    except:
        pass
    
    # Default to current directory with warning
    click.echo(click.style("Warning: Could not determine project root. Using current directory.", fg='yellow'))
    return current_dir

def load_config() -> Dict[str, Any]:
    """Load CLI configuration"""
    config_dir = Path.home() / ".mcp"
    config_file = config_dir / "config.json"
    
    default_config = {
        "table_format": DEFAULT_TABLE_FORMAT,
        "host": DEFAULT_HOST,
        "api_port": DEFAULT_API_PORT,
        "ui_port": DEFAULT_UI_PORT,
        "mock_port": DEFAULT_MOCK_PORT,
        "env_file": ".env",
        "debug": False
    }
    
    if not config_file.exists():
        # Create default config
        config_dir.mkdir(exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return default_config

def save_config(config: Dict[str, Any]):
    """Save CLI configuration"""
    config_dir = Path.home() / ".mcp"
    config_file = config_dir / "config.json"
    
    config_dir.mkdir(exist_ok=True)
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def format_table_data(data: Union[List[Dict], pd.DataFrame], format: str = DEFAULT_TABLE_FORMAT) -> str:
    """Format table data in various formats"""
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a list of dictionaries or a pandas DataFrame")
    
    if format.lower() == "html":
        return df.to_html(index=False)
    elif format.lower() == "csv":
        return df.to_csv(index=False)
    elif format.lower() == "json":
        return json.dumps(json.loads(df.to_json(orient="records")), indent=2)
    elif format.lower() == "table":
        return str(df)
    else:  # Default to markdown
        return df.to_markdown(index=False)

def check_dependencies() -> bool:
    """Check if all dependencies are installed"""
    try:
        # Check Python packages
        import fastapi
        import uvicorn
        import fastmcp
        import chainlit
        import pydantic
        import httpx
        import redis
        
        # Check external tools
        npm_version = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if npm_version.returncode != 0:
            click.echo(click.style("Warning: npm not found. Some features may not work correctly.", fg='yellow'))
        
        return True
    except ImportError as e:
        click.echo(click.style(f"Missing dependency: {e.name}", fg='red'))
        return False

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode with verbose output')
@click.pass_context
def cli(ctx, debug):
    """CRM MCP Server CLI

    This tool helps you manage and run the CRM MCP Server with various options.
    """
    # Initialize context object with configuration
    ctx.ensure_object(dict)
    
    # Load config
    config = load_config()
    if debug:
        config["debug"] = True
    
    # Set up debug logging if enabled
    if debug:
        logger.setLevel(logging.DEBUG)
        click.echo(click.style("Debug mode enabled", fg='yellow'))
    
    ctx.obj = config

@cli.command(name='run')
@click.option('--host', default=None, help='Host to run the server on')
@click.option('--port', default=None, type=int, help='Port to run the server on')
@click.option('--env', default=None, help='Path to environment file')
@click.option('--no-ui', is_flag=True, help='Run only the API server without the UI')
@click.option('--mode', type=click.Choice(['http', 'stdio']), default='http', 
              help='Server mode: http or stdio')
@click.pass_context
def run_server(ctx, host, port, env, no_ui, mode):
    """Run the CRM MCP Server with Chainlit UI"""
    # Get configuration with command-line overrides
    config = ctx.obj
    host = host or config.get('host', DEFAULT_HOST)
    port = port or config.get('api_port', DEFAULT_API_PORT)
    env_file = env or config.get('env_file', '.env')
    
    try:
        # Ensure we're in the project root
        project_root = get_project_root()
        os.chdir(project_root)

        # Load environment variables
        if os.path.exists(env_file):
            from dotenv import load_dotenv
            load_dotenv(env_file)
            click.echo(click.style(f"✓ Loaded environment from {env_file}", fg='green'))
        else:
            click.echo(click.style(f"Warning: Environment file {env_file} not found", fg='yellow'))

        if mode == 'stdio':
            # Run in stdio mode
            click.echo(click.style("Starting MCP server in stdio mode...", fg='yellow'))
            
            try:
                from app.streamlined_mcp_server import mcp, process_message, run_stdio_mode
                import asyncio
                
                # Run the MCP server in STDIO mode
                asyncio.run(run_stdio_mode(mcp, process_message))
            except ImportError:
                # Fall back to original mcp_server
                click.echo(click.style("Streamlined server not found, falling back to original MCP server", fg='yellow'))
                from app.mcp_server import mcp, process_message, run_stdio_mode
                import asyncio
                
                # Run the MCP server in STDIO mode
                asyncio.run(run_stdio_mode(mcp, process_message))
            
            return
            
        # Print initial server URLs
        if not no_ui:
            print_server_urls(host, port)

        # Start the FastAPI server in the background
        try:
            # Try to use the streamlined server first
            server_cmd = [
                "uvicorn",
                "app.streamlined_mcp_server:app",
                "--host", host,
                "--port", str(port),
                "--reload"
            ]
            
            click.echo(click.style("Starting Streamlined FastAPI server...", fg='yellow'))
        except ImportError:
            # Fall back to original server
            server_cmd = [
                "uvicorn",
                "app.mcp_server:main_factory",
                "--host", host,
                "--port", str(port),
                "--reload",
                "--factory"
            ]
            
            click.echo(click.style("Starting FastAPI server...", fg='yellow'))

        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Start Chainlit if UI is enabled
        chainlit_process = None
        if not no_ui:
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

        # Monitor processes
        try:
            # Wait for servers to start
            time.sleep(2)

            # Track server readiness
            api_ready = False
            chainlit_ready = not chainlit_process  # If no UI, consider it ready

            while True:
                # Check server status
                if server_process.poll() is not None:
                    click.echo(click.style("❌ FastAPI server stopped unexpectedly", fg='red'))
                    break

                if chainlit_process and chainlit_process.poll() is not None:
                    click.echo(click.style("❌ Chainlit stopped unexpectedly", fg='red'))
                    break

                # Process FastAPI output
                api_output = server_process.stdout.readline()
                if api_output:
                    if "Application startup complete" in api_output and not api_ready:
                        api_ready = True
                        click.echo(click.style("✓ FastAPI server is ready!", fg='green'))
                    click.echo(f"FastAPI: {api_output.strip()}")

                # Process Chainlit output if UI is enabled
                if chainlit_process:
                    chainlit_output = chainlit_process.stdout.readline()
                    if chainlit_output:
                        if "Chainlit app running" in chainlit_output and not chainlit_ready:
                            chainlit_ready = True
                            click.echo(click.style("✓ Chainlit UI is ready!", fg='green'))
                        click.echo(f"Chainlit: {chainlit_output.strip()}")

                # If both servers are ready, print URLs one final time
                if api_ready and chainlit_ready and not hasattr(run_server, 'urls_printed'):
                    if not no_ui:
                        print_server_urls(host, port, style="ready")
                    else:
                        click.echo("\n" + "-" * 60)
                        click.echo("✨ Server is ready!")
                        click.echo(f"👉 API Server: {click.style(f'http://{host}:{port}', fg='green', bold=True)}")
                        click.echo("Press Ctrl+C to stop the server")
                        click.echo("-" * 60 + "\n")
                    run_server.urls_printed = True

        except KeyboardInterrupt:
            click.echo("\n" + click.style("Shutting down servers...", fg='yellow'))
            server_process.terminate()
            if chainlit_process:
                chainlit_process.terminate()
            server_process.wait()
            if chainlit_process:
                chainlit_process.wait()
            click.echo(click.style("✓ Servers stopped", fg='green'))

    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=config.get("debug", False))
        click.echo(click.style(f"Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='init')
@click.option('--force', is_flag=True, help='Force initialization even if directories exist')
@click.pass_context
def init_project(ctx, force):
    """Initialize the CRM MCP Server project"""
    try:
        # Create necessary directories
        dirs = [
            "app/tools",
            "app/utils",
            "app/registry",
            "tests",
            "project-docs",
            "logs",
            "output"
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            click.echo(click.style(f"✓ Created directory: {dir_path}", fg='green'))

        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists() or force:
            env_file.write_text("""# CRM MCP Server Environment Variables
SERVER_NAME="CRM MCP Server"
SERVER_DESCRIPTION="MCP Server for CRM information and financial tools"
API_PREFIX="/api/v1"
CORS_ORIGINS=["http://localhost:8000", "http://localhost:8001"]
VERSION="0.1.0"
ENVIRONMENT="development"
LOG_LEVEL="INFO"
MCP_SERVER_HOST="0.0.0.0"
MCP_SERVER_PORT=8080
""")
            click.echo(click.style("✓ Created .env file", fg='green'))

        # Create config directory
        config_dir = Path.home() / ".mcp"
        config_dir.mkdir(exist_ok=True)
        
        # Save current config
        save_config(ctx.obj)
        
        click.echo(click.style("✓ Project initialized successfully!", fg='green'))

    except Exception as e:
        logger.error(f"Error initializing project: {e}", exc_info=ctx.obj.get("debug", False))
        click.echo(click.style(f"Error initializing project: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='install')
@click.option('--npm/--no-npm', default=True, help='Install npm dependencies')
@click.option('--python/--no-python', default=True, help='Install Python dependencies')
@click.option('--upgrade', is_flag=True, help='Upgrade existing packages')
@click.pass_context
def install_dependencies(ctx, npm, python, upgrade):
    """Install MCP server and all dependencies"""
    try:
        project_root = get_project_root()
        os.chdir(project_root)
        
        if python:
            click.echo(click.style("Installing Python dependencies...", fg='yellow'))
            
            # Install Python dependencies
            pip_cmd = [sys.executable, "-m", "pip", "install"]
            
            if upgrade:
                pip_cmd.append("--upgrade")
                
            pip_cmd.extend(["-e", "."])
            
            result = subprocess.run(pip_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                click.echo(click.style("✓ Python dependencies installed successfully", fg='green'))
            else:
                click.echo(click.style(f"Error installing Python dependencies: {result.stderr}", fg='red'))
                
            # Install additional dependencies
            if Path("requirements-extra.txt").exists():
                click.echo(click.style("Installing additional Python dependencies...", fg='yellow'))
                pip_cmd = [sys.executable, "-m", "pip", "install"]
                
                if upgrade:
                    pip_cmd.append("--upgrade")
                    
                pip_cmd.extend(["-r", "requirements-extra.txt"])
                
                result = subprocess.run(pip_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    click.echo(click.style("✓ Additional Python dependencies installed successfully", fg='green'))
                else:
                    click.echo(click.style(f"Error installing additional Python dependencies: {result.stderr}", fg='red'))
        
        if npm:
            # Check if package.json exists
            if Path("package.json").exists():
                click.echo(click.style("Installing npm dependencies...", fg='yellow'))
                
                npm_cmd = ["npm", "install"]
                if upgrade:
                    npm_cmd = ["npm", "update"]
                    
                result = subprocess.run(npm_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    click.echo(click.style("✓ npm dependencies installed successfully", fg='green'))
                else:
                    click.echo(click.style(f"Error installing npm dependencies: {result.stderr}", fg='red'))
            else:
                click.echo(click.style("No package.json found, skipping npm dependencies", fg='yellow'))
        
        # Check dependencies
        if check_dependencies():
            click.echo(click.style("✓ All dependencies installed successfully", fg='green'))
        else:
            click.echo(click.style("Some dependencies are missing. Please check the output above.", fg='yellow'))
            
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}", exc_info=ctx.obj.get("debug", False))
        click.echo(click.style(f"Error installing dependencies: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='update')
@click.pass_context
def update_mcp(ctx):
    """Update MCP server to the latest version"""
    ctx.invoke(install_dependencies, npm=True, python=True, upgrade=True)

@cli.command(name='status')
@click.pass_context
def check_status(ctx):
    """Check status of running MCP server instances"""
    try:
        import psutil
        import httpx
        
        # Look for running MCP processes
        mcp_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any(x in ' '.join(cmdline) for x in ['mcp_server', 'streamlined_mcp_server']):
                    mcp_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if not mcp_processes:
            click.echo(click.style("No MCP server processes found", fg='yellow'))
            return
        
        click.echo(click.style("Running MCP server processes:", fg='blue'))
        for proc in mcp_processes:
            click.echo(f"PID: {proc.pid}, Command: {' '.join(proc.info['cmdline'])}")
        
        # Try to connect to known ports
        ports_to_check = [8000, 8080, 8001, 8081]
        click.echo(click.style("\nChecking server endpoints:", fg='blue'))
        
        for port in ports_to_check:
            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(f"http://localhost:{port}/ping")
                    if response.status_code == 200:
                        click.echo(click.style(f"✓ Server running on port {port}: {response.json()}", fg='green'))
                    else:
                        click.echo(f"Port {port}: HTTP {response.status_code}")
            except Exception:
                pass
                
    except ImportError:
        click.echo(click.style("psutil and/or httpx package not installed. Cannot check process status.", fg='yellow'))
    except Exception as e:
        logger.error(f"Error checking status: {e}", exc_info=ctx.obj.get("debug", False))
        click.echo(click.style(f"Error checking status: {str(e)}", fg='red'), err=True)

@cli.command(name='mock')
@click.option('--port', default=None, type=int, help='Port to run the mock server on')
@click.pass_context
def run_mock_server(ctx, port):
    """Run mock financial endpoints for testing"""
    try:
        # Get configuration with command-line overrides
        config = ctx.obj
        port = port or config.get('mock_port', DEFAULT_MOCK_PORT)
        
        click.echo(click.style(f"Starting mock financial server on port {port}...", fg='yellow'))
        
        # Check if dummy_financial_server.py exists
        dummy_server_path = Path("examples/dummy_financial_server.py")
        if not dummy_server_path.exists():
            click.echo(click.style("Mock server file not found. Creating it...", fg='yellow'))
            
            # Create examples directory if it doesn't exist
            Path("examples").mkdir(exist_ok=True)
            
            # Create dummy_financial_server.py
            with open(dummy_server_path, "w") as f:
                f.write("""from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import random

app = FastAPI(title="Dummy Financial Service")

class RequestPayload(BaseModel):
    appCode: str
    values: List[Any]

# In-memory tables
TOP_CLIENTS = []
REVENUE_BY_TIME = []
CLIENT_VALUE_BY_PRODUCT = []

REGIONS = ["CAN", "USA", "EUR", "APAC", "LATAM", "OTHER"]
FOCUS_LISTS = ["Focus40", "FS30", "Corp100"]
PRODUCTS = ["Bonds", "Equities", "FX", "Derivatives", "Commodities"]


def _generate_tables():
    random.seed(0)
    for i in range(150):
        TOP_CLIENTS.append({
            "ClientName": f"Client {i}",
            "ClientCDRID": 1000 + i,
            "RevenueYTD": round(random.uniform(1_000_000, 10_000_000), 2),
            "RegionName": random.choice(REGIONS),
            "FocusList": random.choice(FOCUS_LISTS),
            "InteractionCMOCYTD": random.randint(0, 20),
            "InteractionGMOCYTD": random.randint(0, 20),
            "InteractionYTD": random.randint(0, 40),
            "InteractionCMOCPrevYTD": random.randint(0, 20),
            "InteractionGMOCPrevYTD": random.randint(0, 20),
            "InteractionPrevYTD": random.randint(0, 40),
        })

        REVENUE_BY_TIME.append({
            "ClientName": f"Client {i}",
            "ClientCDRID": 1000 + i,
            "RevenueYTD": round(random.uniform(1_000_000, 10_000_000), 2),
            "RevenuePrevYTD": round(random.uniform(500_000, 9_000_000), 2),
            "InteractionCMOCYTD": random.randint(0, 20),
            "InteractionGMOCYTD": random.randint(0, 20),
            "InteractionYTD": random.randint(0, 40),
            "TimePeriodList": [2023, 2024, 2025],
            "TimePeriodCategory": random.choice(["FY", "CY"]),
        })

        CLIENT_VALUE_BY_PRODUCT.append({
            "ProductName": random.choice(PRODUCTS),
            "RevenueYTD": round(random.uniform(500_000, 5_000_000), 2),
            "RevenuePrevYTD": round(random.uniform(500_000, 5_000_000), 2),
            "ProductID": 2000 + i,
            "ProductHierarchyDepth": random.randint(1, 3),
            "ParentProductID": random.randint(1000, 1999),
            "TimePeriodList": [2023, 2024, 2025],
        })

_generate_tables()

@app.post("/procedure/memsql__client1__getTopClients")
def get_top_clients(_: RequestPayload):
    return {"status": "success", "data": TOP_CLIENTS}

@app.post("/procedure/memsql__client1__getRevenueTotalByTimePeriod")
def get_revenue_by_time(_: RequestPayload):
    return {"status": "success", "data": REVENUE_BY_TIME}

@app.post("/procedure/memsql__client1__getClientValueRevenueByProduct")
def get_client_value_by_product(_: RequestPayload):
    return {"status": "success", "data": CLIENT_VALUE_BY_PRODUCT}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
""")
            click.echo(click.style("✓ Created mock server file", fg='green'))
        
        # Run the mock server
        mock_cmd = [
            sys.executable,
            str(dummy_server_path),
            "--port", str(port)
        ]
        
        # Run in foreground
        click.echo(click.style(f"Mock financial server running at http://localhost:{port}", fg='green'))
        click.echo(click.style("Press Ctrl+C to stop the server", fg='yellow'))
        
        subprocess.run(mock_cmd)
        
    except Exception as e:
        logger.error(f"Error running mock server: {e}", exc_info=ctx.obj.get("debug", False))
        click.echo(click.style(f"Error running mock server: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='format-table')
@click.argument('input_file', type=click.Path(exists=True), required=False)
@click.option('--format', type=click.Choice(VALID_TABLE_FORMATS), default=None, 
              help='Output format (markdown, html, csv, json, table)')
@click.option('--output', type=click.Path(), help='Output file (default: stdout)')
@click.pass_context
def format_table(ctx, input_file, format, output):
    """Format table data from file or stdin"""
    try:
        # Get format from config if not specified
        config = ctx.obj
        format = format or config.get('table_format', DEFAULT_TABLE_FORMAT)
        
        # Read input data
        if input_file:
            # Determine file type from extension
            file_path = Path(input_file)
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.csv':
                data = pd.read_csv(file_path)
            elif file_ext == '.json':
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    data = pd.DataFrame(json_data)
                elif isinstance(json_data, dict) and 'data' in json_data:
                    # Handle common API response format
                    data = pd.DataFrame(json_data['data'])
                else:
                    click.echo(click.style("Error: JSON data must be a list of objects or have a 'data' field", fg='red'))
                    sys.exit(1)
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                click.echo(click.style(f"Unsupported file type: {file_ext}", fg='red'))
                sys.exit(1)
        else:
            # Read from stdin
            try:
                stdin_data = sys.stdin.read().strip()
                if not stdin_data:
                    click.echo(click.style("Error: No input data provided", fg='red'))
                    sys.exit(1)
                
                # Try to parse as JSON first
                try:
                    json_data = json.loads(stdin_data)
                    if isinstance(json_data, list):
                        data = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict) and 'data' in json_data:
                        data = pd.DataFrame(json_data['data'])
                    else:
                        click.echo(click.style("Error: JSON data must be a list of objects or have a 'data' field", fg='red'))
                        sys.exit(1)
                except json.JSONDecodeError:
                    # Try to parse as CSV
                    try:
                        import io
                        data = pd.read_csv(io.StringIO(stdin_data))
                    except Exception:
                        click.echo(click.style("Error: Could not parse input as JSON or CSV", fg='red'))
                        sys.exit(1)
            except Exception as e:
                click.echo(click.style(f"Error reading from stdin: {e}", fg='red'))
                sys.exit(1)
        
        # Format the data
        formatted_data = format_table_data(data, format)
        
        # Output the formatted data
        if output:
            with open(output, 'w') as f:
                f.write(formatted_data)
            click.echo(click.style(f"✓ Formatted table written to {output}", fg='green'))
        else:
            click.echo(formatted_data)
            
    except Exception as e:
        logger.error(f"Error formatting table: {e}", exc_info=ctx.obj.get("debug", False))
        click.echo(click.style(f"Error formatting table: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='config')
@click.option('--table-format', type=click.Choice(VALID_TABLE_FORMATS), 
              help='Set default table format')
@click.option('--host', help='Set default host')
@click.option('--api-port', type=int, help='Set default API port')
@click.option('--ui-port', type=int, help='Set default UI port')
@click.option('--env-file', help='Set default environment file')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--reset', is_flag=True, help='Reset configuration to defaults')
@click.pass_context
def configure(ctx, table_format, host, api_port, ui_port, env_file, show, reset):
    """Configure CLI settings"""
    try:
        config = ctx.obj
        
        if reset:
            # Reset to defaults
            config = {
                "table_format": DEFAULT_TABLE_FORMAT,
                "host": DEFAULT_HOST,
                "api_port": DEFAULT_API_PORT,
                "ui_port": DEFAULT_UI_PORT,
                "mock_port": DEFAULT_MOCK_PORT,
                "env_file": ".env",
                "debug": False
            }
            click.echo(click.style("Configuration reset to defaults", fg='green'))
        else:
            # Update configuration with provided values
            if table_format:
                config["table_format"] = table_format
            if host:
                config["host"] = host
            if api_port:
                config["api_port"] = api_port
            if ui_port:
                config["ui_port"] = ui_port
            if env_file:
                config["env_file"] = env_file
        
        # Save configuration
        save_config(config)
        
        # Show current configuration
        if show or not any([table_format, host, api_port, ui_port, env_file, reset]):
            click.echo(click.style("Current configuration:", fg='blue'))
            for key, value in config.items():
                click.echo(f"{key}: {value}")
                
    except Exception as e:
        logger.error(f"Error configuring CLI: {e}", exc_info=ctx.obj.get("debug", False))
        click.echo(click.style(f"Error configuring CLI: {str(e)}", fg='red'), err=True)
        sys.exit(1)

@cli.command(name='crm-mcp')
@click.pass_context
def crm_mcp(ctx):
    """Run the CRM MCP server in stdio mode for Chainlit integration"""
    ctx.invoke(run_server, host=None, port=None, env=None, no_ui=True, mode='stdio')

if __name__ == '__main__':
    cli(obj={})
