import pandas as pd
from typing import Dict, Any, Optional
from .storage import AzureStorageBackend
import yaml
import os

class DataLayer:
    def __init__(self):
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize storage backend
        self.storage = AzureStorageBackend(
            connection_string=self.config['storage']['azure']['connection_string'],
            container=self.config['storage']['azure']['container']
        )
        
        # Load data mappings
        self.data_mappings = self.config['storage']['azure']['data_mappings']
        self.metadata = self.config['metadata']['data_descriptions']
    
    async def get_data(self, dataset_name: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Get data from a specific dataset."""
        if dataset_name not in self.data_mappings:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        file_path = self.data_mappings[dataset_name]
        metadata = self.metadata[dataset_name]
        
        # Read the file content
        content = await self.storage.read_file(file_path)
        
        # Convert to pandas DataFrame
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(content)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Apply query if provided
        if query:
            df = df.query(query)
        
        return {
            "data": df.to_dict(orient='records'),
            "metadata": metadata,
            "total_rows": len(df),
            "columns": list(df.columns)
        }
    
    async def get_jobs_data(self) -> Dict[str, Any]:
        """Get jobs statistics."""
        return await self.get_data('jobs_data')
    
    async def get_employee_data(self) -> Dict[str, Any]:
        """Get employee data."""
        return await self.get_data('employee_data')
    
    async def get_sales_data(self) -> Dict[str, Any]:
        """Get sales data."""
        return await self.get_data('sales_data') 