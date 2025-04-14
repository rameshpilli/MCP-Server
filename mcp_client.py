#!/usr/bin/env python
"""
MCP Client - Python SDK for the Model Context Protocol

This client library simplifies integration with the MCP server,
providing a clean interface for accessing models and data sources.
"""

import json
import asyncio
import logging
from typing import Dict, List, Union, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    pass

class AuthenticationError(MCPClientError):
    """Raised when authentication fails."""
    pass

class ResourceNotFoundError(MCPClientError):
    """Raised when a requested resource is not found."""
    pass

class MCPClient:
    """Client for the Model Context Protocol (MCP) server."""
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "http://localhost:8000/mcp",
        timeout: int = 30,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the MCP client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the MCP server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger("mcp_client")
        
        # Configure client headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "mcp-python-client/1.0.0"
        }
        
        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=timeout,
            headers=self.headers,
            follow_redirects=True
        )
        
        # Initialize async HTTP client
        self.async_client = httpx.AsyncClient(
            timeout=timeout,
            headers=self.headers,
            follow_redirects=True
        )
    
    def __del__(self):
        """Cleanup when the client is destroyed."""
        try:
            self.client.close()
        except:
            pass
        
        try:
            asyncio.create_task(self.async_client.aclose())
        except:
            pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    def _request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the MCP server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            AuthenticationError: When API key is invalid
            ResourceNotFoundError: When resource is not found
            MCPClientError: For other client errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.client.get(url)
            elif method.upper() == "POST":
                response = self.client.post(url, json=data)
            else:
                raise MCPClientError(f"Unsupported HTTP method: {method}")
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found: {endpoint}")
            elif response.status_code >= 400:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                raise MCPClientError(f"API error ({response.status_code}): {error_data.get('error', 'Unknown error')}")
            
            return response.json()
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error during {method} request to {url}: {str(e)}")
            raise MCPClientError(f"HTTP error: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    async def _async_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an async request to the MCP server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            AuthenticationError: When API key is invalid
            ResourceNotFoundError: When resource is not found
            MCPClientError: For other client errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = await self.async_client.get(url)
            elif method.upper() == "POST":
                response = await self.async_client.post(url, json=data)
            else:
                raise MCPClientError(f"Unsupported HTTP method: {method}")
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found: {endpoint}")
            elif response.status_code >= 400:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                raise MCPClientError(f"API error ({response.status_code}): {error_data.get('error', 'Unknown error')}")
            
            return response.json()
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error during async {method} request to {url}: {str(e)}")
            raise MCPClientError(f"HTTP error: {str(e)}")
    
    # ----------------
    # Model Operations
    # ----------------
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information
        """
        response = self._request("GET", "resources/models://list")
        return response.get("data", [])
    
    async def async_list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models asynchronously.
        
        Returns:
            List of model information
        """
        response = await self._async_request("GET", "resources/models://list")
        return response.get("data", [])
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get details for a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model details
        """
        response = self._request("GET", f"resources/models://{model_id}")
        return response.get("data", {})
    
    async def async_get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get details for a specific model asynchronously.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model details
        """
        response = await self._async_request("GET", f"resources/models://{model_id}")
        return response.get("data", {})
    
    def generate_with_model(self, model_id: str, prompt: str, options: Dict[str, Any] = None) -> str:
        """
        Generate text using a specific model.
        
        Args:
            model_id: ID of the model
            prompt: Input prompt for generation
            options: Additional options for generation
            
        Returns:
            Generated text
        """
        data = {
            "model_id": model_id,
            "prompt": prompt,
            **(options or {})
        }
        response = self._request("POST", "tools/generate_with_model", data=data)
        return response.get("result", "")
    
    async def async_generate_with_model(self, model_id: str, prompt: str, options: Dict[str, Any] = None) -> str:
        """
        Generate text using a specific model asynchronously.
        
        Args:
            model_id: ID of the model
            prompt: Input prompt for generation
            options: Additional options for generation
            
        Returns:
            Generated text
        """
        data = {
            "model_id": model_id,
            "prompt": prompt,
            **(options or {})
        }
        response = await self._async_request("POST", "tools/generate_with_model", data=data)
        return response.get("result", "")
    
    def register_model(
        self, 
        model_id: str, 
        name: str, 
        backend: str, 
        description: str = None, 
        api_base: str = None, 
        version: str = None
    ) -> Dict[str, Any]:
        """
        Register a new model with the model registry.
        
        Args:
            model_id: Unique identifier for the model
            name: Display name for the model
            backend: Backend type (openai, anthropic, etc.)
            description: Optional description
            api_base: Optional API base URL
            version: Optional version string
            
        Returns:
            Registration result with API key
        """
        data = {
            "model_id": model_id,
            "name": name,
            "backend": backend,
            "description": description,
            "api_base": api_base,
            "version": version
        }
        # Filter out None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = self._request("POST", "tools/register_model", data=data)
        return response
    
    # ----------------
    # Data Source Operations
    # ----------------
    
    def list_data_sources(self) -> List[Dict[str, Any]]:
        """
        List all available data sources.
        
        Returns:
            List of data source information
        """
        response = self._request("GET", "resources/sources://list")
        return response.get("data", [])
    
    def query_snowflake(self, source_name: str, query: str) -> List[Dict[str, Any]]:
        """
        Execute a query against a Snowflake data source.
        
        Args:
            source_name: Name of the Snowflake data source
            query: SQL query to execute
            
        Returns:
            Query results as a list of records
        """
        data = {
            "source_name": source_name,
            "query": query
        }
        response = self._request("POST", "tools/query_snowflake", data=data)
        return response.get("result", [])
    
    async def async_query_snowflake(self, source_name: str, query: str) -> List[Dict[str, Any]]:
        """
        Execute a query against a Snowflake data source asynchronously.
        
        Args:
            source_name: Name of the Snowflake data source
            query: SQL query to execute
            
        Returns:
            Query results as a list of records
        """
        data = {
            "source_name": source_name,
            "query": query
        }
        response = await self._async_request("POST", "tools/query_snowflake", data=data)
        return response.get("result", [])
    
    def list_storage_files(self, source_name: str, path: str = "") -> List[Dict[str, Any]]:
        """
        List files in a storage path.
        
        Args:
            source_name: Name of the storage source
            path: Path within the storage
            
        Returns:
            List of file information
        """
        data = {
            "source_name": source_name,
            "path": path
        }
        response = self._request("POST", "tools/list_storage_files", data=data)
        return response.get("result", [])
    
    def get_snowflake_data(self, source_name: str, path: str) -> str:
        """
        Get data from a Snowflake data source.
        
        Args:
            source_name: Name of the Snowflake source
            path: Path to the data (database/schema/table)
            
        Returns:
            Data as a string (usually CSV)
        """
        response = self._request("GET", f"resources/snowflake://{source_name}/{path}")
        return response.get("data", "")
    
    def get_azure_data(self, source_name: str, path: str) -> str:
        """
        Get data from an Azure Blob Storage source.
        
        Args:
            source_name: Name of the Azure source
            path: Path to the data
            
        Returns:
            Data as a string
        """
        response = self._request("GET", f"resources/azure://{source_name}/{path}")
        return response.get("data", "")
    
    def get_s3_data(self, source_name: str, path: str) -> str:
        """
        Get data from an S3 storage source.
        
        Args:
            source_name: Name of the S3 source
            path: Path to the data
            
        Returns:
            Data as a string
        """
        response = self._request("GET", f"resources/s3://{source_name}/{path}")
        return response.get("data", "")
    
    # ----------------
    # Prompt Operations
    # ----------------
    
    def create_data_analysis_prompt(self, data: str, question: str = None) -> str:
        """
        Create a prompt for data analysis.
        
        Args:
            data: Data to analyze
            question: Optional specific question to answer
            
        Returns:
            Formatted prompt text
        """
        data = {
            "data": data,
            "question": question
        }
        # Filter out None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = self._request("POST", "prompts/data_analysis_prompt", data=data)
        return response.get("prompt", "")
    
    def create_query_generator_prompt(self, table_description: str, question: str) -> str:
        """
        Generate a prompt for SQL query generation.
        
        Args:
            table_description: Description of table schema
            question: Natural language question
            
        Returns:
            Formatted prompt text
        """
        data = {
            "table_description": table_description,
            "question": question
        }
        response = self._request("POST", "prompts/query_generator_prompt", data=data)
        return response.get("prompt", "")


if __name__ == "__main__":
    # Example usage
    client = MCPClient(
        api_key="your_api_key_here", 
        base_url="http://localhost:8000/mcp"
    )
    
    # List models
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Generate with a model
    response = client.generate_with_model(
        model_id="gpt4-turbo",
        prompt="What's the weather like today?"
    )
    print(f"Model response: {response}") 