from abc import ABC, abstractmethod
from typing import List, Dict, Any, BinaryIO, Optional
from pathlib import Path
import os
import boto3
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
import json

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def list_files(self, path: str) -> List[Dict[str, Any]]:
        """List files in a directory."""
        pass
    
    @abstractmethod
    def read_file(self, path: str) -> bytes:
        """Read file contents."""
        pass
    
    @abstractmethod
    def write_file(self, path: str, content: bytes) -> bool:
        """Write content to a file."""
        pass
    
    @abstractmethod
    def delete_file(self, path: str) -> bool:
        """Delete a file."""
        pass
    
    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass

class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        full_path = (self.base_path / path).resolve()
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError("Access to path outside base directory is forbidden")
        return full_path
    
    def list_files(self, path: str = ".") -> List[Dict[str, Any]]:
        full_path = self._get_full_path(path)
        if not full_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        files = []
        for item in full_path.iterdir():
            stat = item.stat()
            files.append({
                "name": item.name,
                "path": str(item.relative_to(self.base_path)),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_dir": item.is_dir()
            })
        return files
    
    def read_file(self, path: str) -> bytes:
        full_path = self._get_full_path(path)
        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return full_path.read_bytes()
    
    def write_file(self, path: str, content: bytes) -> bool:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
        return True
    
    def delete_file(self, path: str) -> bool:
        full_path = self._get_full_path(path)
        if full_path.exists():
            full_path.unlink()
            return True
        return False
    
    def file_exists(self, path: str) -> bool:
        try:
            return self._get_full_path(path).exists()
        except ValueError:
            return False

class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend."""
    
    def __init__(self, bucket: str, endpoint_url: Optional[str] = None, **kwargs):
        self.bucket = bucket
        self.s3 = boto3.client('s3', endpoint_url=endpoint_url, **kwargs)
    
    def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        path = path.rstrip("/")
        if path:
            path = f"{path}/"
            
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=path)
        files = []
        
        for item in response.get('Contents', []):
            name = item['Key']
            if path:
                name = name[len(path):]
            
            files.append({
                "name": name,
                "path": item['Key'],
                "size": item['Size'],
                "modified": item['LastModified'].timestamp(),
                "is_dir": name.endswith('/')
            })
        return files
    
    def read_file(self, path: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=path)
        return response['Body'].read()
    
    def write_file(self, path: str, content: bytes) -> bool:
        try:
            self.s3.put_object(Bucket=self.bucket, Key=path, Body=content)
            return True
        except Exception:
            return False
    
    def delete_file(self, path: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False
    
    def file_exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False

class AzureStorageBackend(StorageBackend):
    """Azure Data Lake Storage Gen2 backend."""
    
    def __init__(self, connection_string: str, container: str):
        self.service_client = DataLakeServiceClient.from_connection_string(connection_string)
        self.container = container
        self.filesystem_client = self.service_client.get_file_system_client(container)
    
    def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        path = path.rstrip("/")
        if path:
            path = f"{path}/"
            
        paths = self.filesystem_client.get_paths(path=path)
        files = []
        
        for item in paths:
            name = item.name
            if path:
                name = name[len(path):]
                
            files.append({
                "name": name,
                "path": item.name,
                "size": item.content_length,
                "modified": item.last_modified.timestamp(),
                "is_dir": not item.is_file
            })
        return files
    
    def read_file(self, path: str) -> bytes:
        file_client = self.filesystem_client.get_file_client(path)
        download = file_client.download_file()
        return download.readall()
    
    def write_file(self, path: str, content: bytes) -> bool:
        try:
            file_client = self.filesystem_client.create_file(path)
            file_client.upload_data(content, overwrite=True)
            return True
        except Exception:
            return False
    
    def delete_file(self, path: str) -> bool:
        try:
            file_client = self.filesystem_client.get_file_client(path)
            file_client.delete_file()
            return True
        except Exception:
            return False
    
    def file_exists(self, path: str) -> bool:
        try:
            file_client = self.filesystem_client.get_file_client(path)
            file_client.get_file_properties()
            return True
        except Exception:
            return False 