#!/usr/bin/env python3
"""
MCP Server Launcher

This script starts both the MCP server and the FastAPI server.
It works in both local development and container environments.
"""
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import socket
from pathlib import Path
import threading
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mcp_launcher")

# Set working directory to project root
project_root = Path(__file__).parent.absolute()
os.chdir(project_root)

# Import config after loading environment variables
sys.path.append(str(project_root))
from app.config import config

# Process handles
mcp_server_process = None
api_server_process = None

# Detect if we're running in Kubernetes
IN_KUBERNETES = os.getenv("KUBERNETES_SERVICE_HOST") is not None

def is_port_in_use(host, port):
    """Check if a port is in use"""
    if host == "0.0.0.0":
        # When binding to all interfaces, check localhost
        host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def wait_for_port(host, port, timeout=30):
    """Wait for a port to become available"""
    if host == "0.0.0.0":
        # When binding to all interfaces, check localhost
        host = "127.0.0.1"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.5)
    return False

def start_mcp_server():
    """Start the MCP server with automatic port selection"""
    mcp_host = config.MCP_SERVER_HOST
    mcp_port = config.MCP_SERVER_PORT

    # In Kubernetes, we don't want to check ports or use automatic port selection
    if not IN_KUBERNETES:
        # Check if port is already in use
        if is_port_in_use(mcp_host, mcp_port):
            logger.warning(f"Port {mcp_port} already in use. MCP server will automatically select an available port.")

    logger.info(f"Starting MCP server (configured port: {mcp_port})...")
    env = os.environ.copy()

    # Ensure the transport is set
    if "TRANSPORT" not in env:
        env["TRANSPORT"] = "sse"
    
    # In Kubernetes, bind to all interfaces
    if IN_KUBERNETES:
        env["MCP_SERVER_HOST"] = "0.0.0.0"

    # Create a pipe for real-time output
    read_pipe, write_pipe = os.pipe()
    
    # Start the process with the write end of the pipe
    process = subprocess.Popen(
        [sys.executable, "-m", "app.mcp_server"],
        stdout=write_pipe,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
        bufsize=1  # Line buffered
    )
    
    # Close the write end in the parent
    os.close(write_pipe)
    
    # Create a non-blocking file object from the read end
    import fcntl
    fcntl.fcntl(read_pipe, fcntl.F_SETFL, os.O_NONBLOCK)
    
    # Monitor the process output in real-time
    start_time = time.time()
    timeout = 30  # Wait up to 30 seconds for server to start
    server_ready = False
    
    while time.time() - start_time < timeout:
        try:
            # Try to read output
            output = os.read(read_pipe, 1024).decode()
            if output:
                logger.info(f"[MCP Server] {output.strip()}")
                if "Starting MCP server on" in output:
                    server_ready = True
                    break
        except BlockingIOError:
            # No output available, check if process is still running
            if process.poll() is not None:
                # Process exited
                logger.error(f"MCP server exited with code {process.returncode}")
                # Read any remaining output
                while True:
                    try:
                        output = os.read(read_pipe, 1024).decode()
                        if output:
                            logger.error(f"[MCP Server] {output.strip()}")
                    except BlockingIOError:
                        break
                return None
            time.sleep(0.1)
    
    # Close the read end
    os.close(read_pipe)
    
    if not server_ready:
        logger.warning("Could not confirm MCP server is ready, but continuing anyway...")
        if process.poll() is not None:
            logger.error(f"MCP server failed to start (exit code: {process.returncode})")
            return None
    
    return process

def start_api_server():
    """Start the FastAPI server"""
    api_host = config.HOST
    api_port = config.PORT
    
    # In Kubernetes, we don't want to check ports
    if not IN_KUBERNETES:
        # Check if port is already in use
        if is_port_in_use(api_host, api_port):
            logger.warning(f"Port {api_port} already in use. API server may already be running.")
    
    logger.info(f"Starting API server on {api_host}:{api_port}...")
    
    # In Kubernetes, don't use --reload
    reload_flag = [] if IN_KUBERNETES else ["--reload"]
    
    # Create a pipe for real-time output
    read_pipe, write_pipe = os.pipe()
    
    # Start the process with the write end of the pipe
    process = subprocess.Popen(
        [
            "uvicorn", 
            "app.main:app", 
            "--host", api_host, 
            "--port", str(api_port)
        ] + reload_flag,
        stdout=write_pipe,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Close the write end in the parent
    os.close(write_pipe)
    
    # Create a non-blocking file object from the read end
    import fcntl
    fcntl.fcntl(read_pipe, fcntl.F_SETFL, os.O_NONBLOCK)
    
    # Monitor the process output in real-time
    start_time = time.time()
    timeout = 30  # Wait up to 30 seconds for server to start
    server_ready = False
    
    while time.time() - start_time < timeout:
        try:
            # Try to read output
            output = os.read(read_pipe, 1024).decode()
            if output:
                logger.info(f"[API Server] {output.strip()}")
                if "Application startup complete" in output:
                    server_ready = True
                    break
        except BlockingIOError:
            # No output available, check if process is still running
            if process.poll() is not None:
                # Process exited
                logger.error(f"API server exited with code {process.returncode}")
                # Read any remaining output
                while True:
                    try:
                        output = os.read(read_pipe, 1024).decode()
                        if output:
                            logger.error(f"[API Server] {output.strip()}")
                    except BlockingIOError:
                        break
                return None
            time.sleep(0.1)
    
    # Close the read end
    os.close(read_pipe)
    
    if not server_ready:
        logger.warning("Could not confirm API server is ready, but continuing anyway...")
        if process.poll() is not None:
            logger.error(f"API server failed to start (exit code: {process.returncode})")
            return None
    
    return process

def monitor_process(process, name):
    """Monitor a process and log its output"""
    if process is None:
        return
        
    # Create a pipe for real-time output
    read_pipe, write_pipe = os.pipe()
    
    # Create a non-blocking file object from the read end
    import fcntl
    fcntl.fcntl(read_pipe, fcntl.F_SETFL, os.O_NONBLOCK)
    
    # Redirect process output to the write end of the pipe
    process.stdout = os.fdopen(write_pipe, 'w')
    process.stderr = subprocess.STDOUT
    
    while True:
        if process.poll() is not None:
            logger.info(f"{name} exited with code {process.returncode}")
            break
            
        try:
            # Try to read output
            output = os.read(read_pipe, 1024).decode()
            if output:
                for line in output.splitlines():
                    if line.strip():
                        logger.info(f"[{name}] {line.strip()}")
        except BlockingIOError:
            # No output available, sleep briefly
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error monitoring {name}: {e}")
            break
    
    # Clean up
    try:
        os.close(read_pipe)
    except:
        pass
    try:
        os.close(write_pipe)
    except:
        pass

def cleanup(signum=None, frame=None):
    """Clean up all processes"""
    global mcp_server_process, api_server_process
    
    logger.info("Shutting down servers...")
    
    # Stop MCP server
    if mcp_server_process:
        logger.info("Stopping MCP server...")
        mcp_server_process.terminate()
        try:
            mcp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("MCP server didn't shut down cleanly, killing it")
            mcp_server_process.kill()
    
    # Stop API server
    if api_server_process:
        logger.info("Stopping API server...")
        api_server_process.terminate()
        try:
            api_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("API server didn't shut down cleanly, killing it")
            api_server_process.kill()
    
    logger.info("All servers stopped")
    
    # Exit if this was called as a signal handler
    if signum is not None:
        sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Display startup banner
        logger.info("====================================")
        logger.info("MCP Platform Server Launcher")
        if IN_KUBERNETES:
            logger.info(f"Running in Kubernetes environment")
            logger.info(f"Pod Name: {os.getenv('POD_NAME', 'unknown')}")
            logger.info(f"Namespace: {os.getenv('POD_NAMESPACE', 'default')}")
        logger.info(f"MCP Server will run on: {config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}")
        logger.info(f"API Server will run on: {config.HOST}:{config.PORT}")
        logger.info("====================================")
        
        # Start MCP server
        mcp_server_process = start_mcp_server()
        if mcp_server_process is None:
            logger.error("Failed to start MCP server")
            sys.exit(1)
        
        # Start API server
        api_server_process = start_api_server()
        if api_server_process is None:
            logger.error("Failed to start API server")
            sys.exit(1)
        
        # Create monitor threads
        mcp_monitor = threading.Thread(
            target=monitor_process,
            args=(mcp_server_process, "MCP Server"),
            daemon=True
        )
        api_monitor = threading.Thread(
            target=monitor_process,
            args=(api_server_process, "API Server"),
            daemon=True
        )
        
        # Start monitors
        mcp_monitor.start()
        api_monitor.start()
        
        # Show that everything is running
        logger.info("====================================")
        logger.info("All servers started successfully")
        logger.info(f"MCP Server: http://{config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}")
        logger.info(f"API Server: http://{config.HOST}:{config.PORT}")
        logger.info("Press Ctrl+C to stop")
        logger.info("====================================")
        
        # Wait for both processes
        while True:
            if mcp_server_process and mcp_server_process.poll() is not None:
                logger.error(f"MCP server exited unexpectedly with code {mcp_server_process.returncode}")
                break
            if api_server_process and api_server_process.poll() is not None:
                logger.error(f"API server exited unexpectedly with code {api_server_process.returncode}")
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        cleanup()
