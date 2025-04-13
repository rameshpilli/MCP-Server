import requests
import json
from typing import Dict, List, Optional
import os
from datetime import datetime

class TestLLMClient:
    def __init__(self, mcp_url: str = "http://localhost:8000"):
        self.mcp_url = mcp_url
        self.api_key = None
        self.model_id = None

    def register_model(self) -> None:
        """Register our test LLM model with MCP"""
        url = f"{self.mcp_url}/api/v1/models/register"
        
        model_data = {
            "name": "test-llm",
            "version": "0.1",
            "description": "A test LLM for demonstrating MCP integration",
            "model_type": "text-generation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "max_tokens": {"type": "integer", "default": 100}
                },
                "required": ["prompt"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "generated_text": {"type": "string"},
                    "tokens_used": {"type": "integer"}
                }
            }
        }

        response = requests.post(url, json=model_data)
        if response.status_code == 201:
            data = response.json()
            self.model_id = data["model_id"]
            print(f"Model registered successfully with ID: {self.model_id}")
        else:
            raise Exception(f"Failed to register model: {response.text}")

    def get_api_key(self) -> None:
        """Get an API key from MCP"""
        url = f"{self.mcp_url}/api/v1/auth/api-key"
        
        data = {
            "name": "test-llm-client",
            "description": "API key for test LLM client"
        }

        response = requests.post(url, json=data)
        if response.status_code == 201:
            data = response.json()
            self.api_key = data["api_key"]
            print(f"API key obtained successfully")
        else:
            raise Exception(f"Failed to get API key: {response.text}")

    def query_data(self, query: str) -> Dict:
        """Query MCP for data"""
        if not self.api_key:
            raise Exception("No API key available. Call get_api_key() first")

        url = f"{self.mcp_url}/api/v1/data/query"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        data = {
            "query": query,
            "model_id": self.model_id
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to query data: {response.text}")

    def generate_response(self, prompt: str, context_data: Optional[Dict] = None) -> Dict:
        """Generate a response using the test LLM"""
        # This is a mock LLM that uses the context data from MCP
        response = {
            "generated_text": f"Based on the context data: {context_data}\n\nHere's my response to: {prompt}",
            "tokens_used": len(prompt.split())
        }
        return response

def main():
    # Initialize the test LLM client
    client = TestLLMClient()

    try:
        # Get API key
        client.get_api_key()
        print("✓ Obtained API key")

        # Register model
        client.register_model()
        print("✓ Registered model")

        # Example: Query for some cat job data
        query = "SELECT * FROM cat_jobs LIMIT 5"
        context_data = client.query_data(query)
        print("✓ Retrieved context data")

        # Generate a response using the context
        prompt = "What are some interesting cat jobs in the database?"
        response = client.generate_response(prompt, context_data)
        print("\nGenerated Response:")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 