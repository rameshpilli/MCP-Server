#!/usr/bin/env python3
"""
MCP Agent Registration Script

This script registers an agent with the MCP server and manages its lifecycle.
"""

import argparse
import json
import logging
import os
import requests
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import agent modules
from config import config
import tools
import resources
import prompts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(project_root, "agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_agent")

def register_agent():
    """Register the agent with the MCP server"""
    logger.info("Registering agent with MCP server...")
    
    try:
        # Agent details
        agent_data = {
            "name": config.AGENT_NAME,
            "description": config.AGENT_DESCRIPTION,
            "namespace": config.AGENT_NAMESPACE,
            "capabilities": config.AGENT_CAPABILITIES,
            "metadata": {
                "version": config.AGENT_VERSION,
                "author": config.AGENT_AUTHOR,
                "contact": config.AGENT_CONTACT
            }
        }
        
        # Register the agent
        response = requests.post(
            f"{config.MCP_SERVER_URL}/api/v1/agents/register",
            json=agent_data
        )
        response.raise_for_status()
        
        # Save agent ID
        agent_id = response.json()["id"]
        namespace = response.json()["namespace"]
        
        # Save agent ID to file
        with open(os.path.join(project_root, "agent_id.json"), "w") as f:
            json.dump({
                "agent_id": agent_id,
                "namespace": namespace
            }, f)
        
        logger.info(f"Agent registered successfully with ID: {agent_id}")
        logger.info(f"Using namespace: {namespace}")
        
        return agent_id, namespace
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error registering agent: {e}")
        if hasattr(e, "response") and e.response:
            logger.error(f"Response: {e.response.text}")
        sys.exit(1)

def unregister_agent(agent_id):
    """Unregister the agent from the MCP server"""
    logger.info(f"Unregistering agent with ID: {agent_id}")
    
    try:
        response = requests.delete(
            f"{config.MCP_SERVER_URL}/api/v1/agents/{agent_id}"
        )
        response.raise_for_status()
        
        # Remove agent ID file
        if os.path.exists(os.path.join(project_root, "agent_id.json")):
            os.remove(os.path.join(project_root, "agent_id.json"))
        
        logger.info("Agent unregistered successfully")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error unregistering agent: {e}")
        if hasattr(e, "response") and e.response:
            logger.error(f"Response: {e.response.text}")
        sys.exit(1)

def get_agent_info(agent_id):
    """Get information about the registered agent"""
    logger.info(f"Getting information for agent ID: {agent_id}")
    
    try:
        response = requests.get(
            f"{config.MCP_SERVER_URL}/api/v1/agents/{agent_id}"
        )
        response.raise_for_status()
        
        agent_info = response.json()
        
        print("\n=== Agent Information ===")
        print(f"ID: {agent_info['id']}")
        print(f"Name: {agent_info['name']}")
        print(f"Description: {agent_info['description']}")
        print(f"Namespace: {agent_info['namespace']}")
        print(f"Capabilities: {', '.join(agent_info['capabilities'])}")
        
        components = agent_info['components']
        print("\n=== Components ===")
        print(f"Tools: {len(components['tools'])}")
        for name, desc in components['tools'].items():
            print(f"  - {name}: {desc}")
        
        print(f"Resources: {len(components['resources'])}")
        for name, desc in components['resources'].items():
            print(f"  - {name}: {desc}")
        
        print(f"Prompts: {len(components['prompts'])}")
        for name, desc in components['prompts'].items():
            print(f"  - {name}: {desc}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting agent information: {e}")
        if hasattr(e, "response") and e.response:
            logger.error(f"Response: {e.response.text}")
        sys.exit(1)

def load_existing_agent():
    """Load existing agent ID if available"""
    id_file = os.path.join(project_root, "agent_id.json")
    if os.path.exists(id_file):
        try:
            with open(id_file, "r") as f:
                data = json.load(f)
            return data.get("agent_id"), data.get("namespace")
        except Exception as e:
            logger.error(f"Error loading agent ID: {e}")
    return None, None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP Agent Registration Tool")
    parser.add_argument("--register", action="store_true", help="Register the agent")
    parser.add_argument("--unregister", action="store_true", help="Unregister the agent")
    parser.add_argument("--info", action="store_true", help="Show agent information")
    
    args = parser.parse_args()
    
    # Load existing agent if available
    agent_id, namespace = load_existing_agent()
    
    if args.register:
        if agent_id:
            logger.warning(f"Agent already registered with ID: {agent_id}")
            choice = input("Do you want to re-register the agent? (y/n): ")
            if choice.lower() != "y":
                return
            
            # Unregister existing agent
            unregister_agent(agent_id)
        
        # Register new agent
        agent_id, namespace = register_agent()
    
    elif args.unregister:
        if not agent_id:
            logger.error("No agent ID found. Register the agent first.")
            return
        
        unregister_agent(agent_id)
    
    elif args.info:
        if not agent_id:
            logger.error("No agent ID found. Register the agent first.")
            return
        
        get_agent_info(agent_id)
    
    else:
        if agent_id:
            print(f"Agent already registered with ID: {agent_id}")
            print(f"Namespace: {namespace}")
            print("Use --info to show agent information")
            print("Use --unregister to unregister the agent")
        else:
            print("No agent registered. Use --register to register a new agent.")
            print("Use --help for more information.")

if __name__ == "__main__":
    main() 