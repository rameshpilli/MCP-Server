from abc import ABC, abstractmethod
from typing import List, Dict, Any, BinaryIO, Optional
from pathlib import Path
import os
import boto3
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
import json
import snowflake.connector
import pandas as pd
import io

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

class SnowflakeStorageBackend(StorageBackend):
    """Snowflake storage backend for querying data as virtual files."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.conn = snowflake.connector.connect(**connection_params)
    
    def _get_cursor(self):
        # Ensure connection is active
        if not self.conn.is_active():
            self.conn = snowflake.connector.connect(**self.connection_params)
        return self.conn.cursor()
    
    def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        """
        List tables or views in a schema.
        Path format: 'database/schema' or 'database/schema/table'
        """
        parts = path.strip('/').split('/')
        if len(parts) == 2:
            # List tables in schema
            database, schema = parts
            cursor = self._get_cursor()
            cursor.execute(f"SHOW TABLES IN {database}.{schema}")
            tables = cursor.fetchall()
            
            result = []
            for table in tables:
                result.append({
                    "name": table[1],  # Table name
                    "path": f"{database}/{schema}/{table[1]}",
                    "size": 0,  # Size not directly available
                    "modified": 0,  # Modified time not directly available
                    "is_dir": False,
                    "metadata": {
                        "kind": table[0],  # TABLE, VIEW, etc.
                        "database": database,
                        "schema": schema
                    }
                })
            return result
        elif len(parts) == 1:
            # List schemas in database
            database = parts[0]
            cursor = self._get_cursor()
            cursor.execute(f"SHOW SCHEMAS IN {database}")
            schemas = cursor.fetchall()
            
            result = []
            for schema in schemas:
                result.append({
                    "name": schema[1],  # Schema name
                    "path": f"{database}/{schema[1]}",
                    "size": 0,
                    "modified": 0,
                    "is_dir": True,
                    "metadata": {
                        "database": database
                    }
                })
            return result
        else:
            # List databases
            cursor = self._get_cursor()
            cursor.execute("SHOW DATABASES")
            databases = cursor.fetchall()
            
            result = []
            for db in databases:
                result.append({
                    "name": db[1],  # Database name
                    "path": db[1],
                    "size": 0,
                    "modified": 0,
                    "is_dir": True
                })
            return result
    
    def read_file(self, path: str) -> bytes:
        """
        Read table data as CSV bytes.
        Path format: 'database/schema/table'
        """
        parts = path.strip('/').split('/')
        if len(parts) != 3:
            raise ValueError("Path must be in format 'database/schema/table'")
        
        database, schema, table = parts
        cursor = self._get_cursor()
        cursor.execute(f"SELECT * FROM {database}.{schema}.{table} LIMIT 1000")
        
        # Convert to pandas DataFrame and then to CSV
        df = cursor.fetch_pandas_all()
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return csv_buffer.getvalue().encode('utf-8')
    
    def write_file(self, path: str, content: bytes) -> bool:
        """
        Write data to a table (creates or replaces).
        Path format: 'database/schema/table'
        Content should be CSV data.
        """
        parts = path.strip('/').split('/')
        if len(parts) != 3:
            raise ValueError("Path must be in format 'database/schema/table'")
        
        database, schema, table = parts
        
        # Read the CSV content into a DataFrame
        try:
            df = pd.read_csv(io.BytesIO(content))
            cursor = self._get_cursor()
            
            # Check if table exists
            cursor.execute(f"SHOW TABLES LIKE '{table}' IN {database}.{schema}")
            table_exists = len(cursor.fetchall()) > 0
            
            if table_exists:
                # Use write_pandas to overwrite
                success, _, _ = cursor.write_pandas(
                    df, 
                    f"{database}.{schema}.{table}", 
                    overwrite=True
                )
            else:
                # Create table and write data
                success, _, _ = cursor.write_pandas(
                    df, 
                    f"{database}.{schema}.{table}", 
                    auto_create_table=True
                )
                
            return success
        except Exception as e:
            print(f"Error writing to Snowflake: {e}")
            return False
    
    def delete_file(self, path: str) -> bool:
        """
        Delete a table.
        Path format: 'database/schema/table'
        """
        parts = path.strip('/').split('/')
        if len(parts) != 3:
            raise ValueError("Path must be in format 'database/schema/table'")
        
        database, schema, table = parts
        
        try:
            cursor = self._get_cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {database}.{schema}.{table}")
            return True
        except Exception:
            return False
    
    def file_exists(self, path: str) -> bool:
        """
        Check if a table exists.
        Path format: 'database/schema/table'
        """
        parts = path.strip('/').split('/')
        if len(parts) != 3:
            return False
        
        database, schema, table = parts
        
        try:
            cursor = self._get_cursor()
            cursor.execute(f"SHOW TABLES LIKE '{table}' IN {database}.{schema}")
            return len(cursor.fetchall()) > 0
        except Exception:
            return False 