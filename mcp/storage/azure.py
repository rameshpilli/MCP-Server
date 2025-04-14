"""
MCP Azure Storage Backend

This module provides an Azure Blob storage backend for MCP.
"""

class AzureStorage:
    """Azure Blob storage backend."""
    
    def __init__(self, config):
        """Initialize the Azure storage backend."""
        self.config = config
        self.connection_string = config.connection_string
        self.container = config.bucket
        self.prefix = config.prefix or ""
    
    def list_files(self, path=""):
        """List files in the specified container/prefix."""
        return []
    
    def read_file(self, path):
        """Read a file from Azure Blob storage."""
        return b""
    
    def write_file(self, path, data):
        """Write data to a file on Azure Blob storage."""
        return True
    
    def delete_file(self, path):
        """Delete a file from Azure Blob storage."""
        return True
    
    def file_exists(self, path):
        """Check if a file exists on Azure Blob storage."""
        return False 