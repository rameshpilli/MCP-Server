"""
MCP Inspector Utility

Tool for testing and inspecting MCP server endpoints.
"""

import argparse
import requests
import json
import logging
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_inspector")

class MCPInspector:
    """
    Tool for testing and inspecting MCP server endpoints
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        logger.info(f"Initialized MCPInspector with base URL: {base_url}")
    
    def test_health(self) -> Dict[str, Any]:
        """Test the health endpoint"""
        url = f"{self.base_url}/api/v1/health"
        logger.info(f"Testing health endpoint: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Health endpoint response: Status {response.status_code}")
        return result
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        url = f"{self.base_url}/api/v1/agents"
        logger.info(f"Listing agents: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        agents = result.get("agents", [])
        logger.info(f"Found {len(agents)} agents")
        return agents
    
    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """Get details for a specific agent"""
        url = f"{self.base_url}/api/v1/agents/{agent_id}"
        logger.info(f"Getting agent details: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Agent details retrieved for: {result.get('name')}")
        return result
    
    def register_test_agent(self, name: str = "TestAgent") -> Dict[str, Any]:
        """Register a test agent"""
        url = f"{self.base_url}/api/v1/agents/register"
        
        data = {
            "name": name,
            "description": "Test agent for endpoint inspection",
            "namespace": name.lower(),
            "capabilities": ["test", "inspection"],
            "metadata": {
                "version": "1.0.0",
                "author": "MCPInspector"
            }
        }
        
        logger.info(f"Registering test agent: {url}")
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Test agent registered with ID: {result.get('id')}")
        return result
    
    def unregister_agent(self, agent_id: str) -> Dict[str, Any]:
        """Unregister an agent"""
        url = f"{self.base_url}/api/v1/agents/{agent_id}"
        logger.info(f"Unregistering agent: {url}")
        
        response = requests.delete(url)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Agent unregistered: {result.get('message')}")
        return result
    
    def test_chat(self, message: str) -> Dict[str, Any]:
        """Test the chat endpoint"""
        url = f"{self.base_url}/api/v1/chat"
        
        data = {
            "message": message,
            "session_id": "inspector-test-session"
        }
        
        logger.info(f"Testing chat endpoint: {url}")
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Chat response received")
        return result
    
    def run_test_suite(self):
        """Run a full test suite on all endpoints"""
        results = {}
        
        # Test health endpoint
        try:
            results["health_check"] = {
                "status": "success",
                "data": self.test_health()
            }
        except Exception as e:
            results["health_check"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test agent registration
        agent_id = None
        try:
            agent_data = self.register_test_agent()
            agent_id = agent_data.get("id")
            results["agent_registration"] = {
                "status": "success",
                "data": agent_data
            }
        except Exception as e:
            results["agent_registration"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test agent listing
        try:
            results["list_agents"] = {
                "status": "success",
                "data": self.list_agents()
            }
        except Exception as e:
            results["list_agents"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test agent details (if we have an agent ID)
        if agent_id:
            try:
                results["agent_details"] = {
                    "status": "success",
                    "data": self.get_agent_details(agent_id)
                }
            except Exception as e:
                results["agent_details"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Test chat endpoint
        try:
            results["chat"] = {
                "status": "success",
                "data": self.test_chat("Hello, this is a test message from MCPInspector")
            }
        except Exception as e:
            results["chat"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Clean up - unregister test agent
        if agent_id:
            try:
                results["agent_unregistration"] = {
                    "status": "success",
                    "data": self.unregister_agent(agent_id)
                }
            except Exception as e:
                results["agent_unregistration"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MCP Server Endpoint Inspector")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for the MCP server API")
    parser.add_argument("--test", action="store_true", help="Run full test suite")
    parser.add_argument("--health", action="store_true", help="Test health endpoint")
    parser.add_argument("--list-agents", action="store_true", help="List all agents")
    parser.add_argument("--register", action="store_true", help="Register a test agent")
    parser.add_argument("--agent-id", help="Agent ID for operations that require it")
    parser.add_argument("--unregister", action="store_true", help="Unregister an agent (requires --agent-id)")
    parser.add_argument("--chat", help="Test chat with a message")
    
    args = parser.parse_args()
    
    inspector = MCPInspector(base_url=args.url)
    
    # Process commands
    if args.test:
        results = inspector.run_test_suite()
        print(json.dumps(results, indent=2))
    
    elif args.health:
        result = inspector.test_health()
        print(json.dumps(result, indent=2))
    
    elif args.list_agents:
        agents = inspector.list_agents()
        print(json.dumps(agents, indent=2))
    
    elif args.register:
        result = inspector.register_test_agent()
        print(json.dumps(result, indent=2))
    
    elif args.unregister:
        if not args.agent_id:
            print("Error: --agent-id is required with --unregister")
            sys.exit(1)
        result = inspector.unregister_agent(args.agent_id)
        print(json.dumps(result, indent=2))
    
    elif args.chat:
        result = inspector.test_chat(args.chat)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 