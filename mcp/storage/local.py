"""
MCP Local Storage Backend

This module provides a local file system storage backend for MCP.
"""

class LocalStorage:
    """Local file system storage backend."""
    
    def __init__(self, config):
        """Initialize the local storage backend."""
        self.config = config
        self.path = config.path or "storage"
    
    def list_files(self, path=""):
        """List files in the specified directory."""
        return []
    
    def read_file(self, path):
        """Read a file from the local file system."""
        return b""
    
    def write_file(self, path, data):
        """Write data to a file on the local file system."""
        return True
    
    def delete_file(self, path):
        """Delete a file from the local file system."""
        return True
    
    def file_exists(self, path):
        """Check if a file exists on the local file system."""
        return False 