#!/usr/bin/env python3
"""
Claude MCP Demo

This script demonstrates how to use Claude with our MCP implementation
to perform model requests and data source operations.

Requirements:
- Claude API key in CLAUDE_API_KEY environment variable
- A running MCP server (python mcp_server.py)

Usage:
    python demo_claude_mcp.py
"""

import os
import sys
import asyncio
import json
from pathlib import Path
import httpx

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check if Claude API key is available
claude_api_key = os.getenv("CLAUDE_API_KEY")
if not claude_api_key:
    print("Error: CLAUDE_API_KEY environment variable is not set.")
    print("Please set your Claude API key in the .env file or environment.")
    sys.exit(1)

# Configure Claude API
CLAUDE_API_BASE = os.getenv("CLAUDE_API_BASE", "https://api.anthropic.com")
CLAUDE_API_VERSION = os.getenv("CLAUDE_API_VERSION", "2023-06-01")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")

# Configure MCP server
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

async def query_claude_with_context(
    prompt: str,
    model: str = CLAUDE_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7
) -> str:
    """
    Send a prompt to Claude with custom context information.
    This allows Claude to access our MCP server resources.
    """
    print(f"Querying Claude with MCP context...")
    
    headers = {
        "x-api-key": claude_api_key,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json",
    }
    
    # Build the messages with system prompt to use MCP
    system_prompt = f"""You are Claude, an AI assistant augmented with MCP (Model Context Protocol) capabilities.
You have access to the following Model Context Protocol server: {MCP_SERVER_URL}/mcp

The server exposes these resources:
- /models - List all available models
- /models/<model_id> - Get information about a specific model
- /sources - List all available data sources
- /snowflake/<source>/<path> - Access data from Snowflake
- /azure/<source>/<path> - Access data from Azure Storage
- /s3/<source>/<path> - Access data from S3

You can also use these tools:
- query_snowflake - Execute SQL queries against Snowflake
- generate_with_model - Generate text using a specific model

Feel free to access these resources as needed to help the user."""
    
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CLAUDE_API_BASE}/v1/messages",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                print(error_message)
                return error_message
    except Exception as e:
        error_message = f"Error calling Claude API: {str(e)}"
        print(error_message)
        return error_message

async def demo_model_registry_query():
    """Demo asking Claude to query our model registry via MCP."""
    prompt = """
    Can you help me understand what AI models are available in my organization's model registry?
    Please list the models, their descriptions, and what they're best suited for.
    """
    
    response = await query_claude_with_context(prompt)
    print("\n=== Claude's Response ===\n")
    print(response)
    print("\n=== End of Response ===\n")

async def demo_data_source_query():
    """Demo asking Claude to analyze data from a Snowflake data source."""
    prompt = """
    I need you to help me analyze our sales data from Snowflake. Please:
    
    1. Check what data sources are available
    2. Look for a Snowflake data source
    3. If you find one, please query it to analyze our most recent sales data
    4. Provide a summary of the data and any insights you find
    """
    
    response = await query_claude_with_context(prompt)
    print("\n=== Claude's Response ===\n")
    print(response)
    print("\n=== End of Response ===\n")

async def demo_model_comparison():
    """Demo asking Claude to compare multiple models from our registry."""
    prompt = """
    I'm working on a new natural language processing application and need to choose the right model.
    Can you help me compare the available models in our registry?
    
    Specifically, I'd like to know:
    1. Which models are best for text summarization
    2. What are the trade-offs in terms of performance and cost
    3. Any specific configuration recommendations for each model
    """
    
    response = await query_claude_with_context(prompt)
    print("\n=== Claude's Response ===\n")
    print(response)
    print("\n=== End of Response ===\n")

async def run_demo():
    """Run the full Claude MCP demo."""
    print("Starting Claude MCP Integration Demo...")
    
    # Check if MCP server is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MCP_SERVER_URL}/health")
            if response.status_code != 200:
                print(f"Warning: MCP server health check failed: {response.status_code}")
                print(f"Make sure the MCP server is running at {MCP_SERVER_URL}")
                proceed = input("Do you want to proceed anyway? (y/n): ")
                if proceed.lower() != 'y':
                    return
    except Exception as e:
        print(f"Warning: Could not connect to MCP server: {str(e)}")
        print(f"Make sure the MCP server is running at {MCP_SERVER_URL}")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Run demos
    demos = {
        "1": ("Model Registry Query", demo_model_registry_query),
        "2": ("Data Source Query", demo_data_source_query),
        "3": ("Model Comparison", demo_model_comparison),
    }
    
    print("\nAvailable demos:")
    for key, (name, _) in demos.items():
        print(f"{key}. {name}")
    print("4. Run all demos")
    print("5. Exit")
    
    choice = input("\nSelect a demo to run (1-5): ")
    
    if choice == "4":
        for name, demo_func in demos.values():
            print(f"\n=== Running Demo: {name} ===")
            await demo_func()
            print(f"=== Demo {name} Completed ===\n")
    elif choice in demos:
        name, demo_func = demos[choice]
        print(f"\n=== Running Demo: {name} ===")
        await demo_func()
        print(f"=== Demo {name} Completed ===\n")
    elif choice == "5":
        print("Exiting demo.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    asyncio.run(run_demo()) 