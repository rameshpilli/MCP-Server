from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
import asyncio
from azure.storage.filedatalake import DataLakeServiceClient
import psycopg2
from psycopg2.extras import Json
import snowflake.connector
from pydantic import BaseModel

class QueryLog(BaseModel):
    model_id: str
    query_text: str
    query_type: str
    parameters: Dict[str, Any]
    response: Dict[str, Any]
    timestamp: datetime
    execution_time: float
    status: str
    error: Optional[str] = None

class QueryLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_type = config.get("storage_type", "postgres")
        self._init_storage()

    def _init_storage(self):
        if self.storage_type == "azure":
            self.datalake_client = DataLakeServiceClient(
                account_url=f"https://{self.config['azure_storage_account']}.dfs.core.windows.net",
                credential=self.config["azure_storage_key"]
            )
            self.container_client = self.datalake_client.get_file_system_client(
                file_system=self.config["azure_container"]
            )
        elif self.storage_type == "postgres":
            self.pg_conn = psycopg2.connect(
                dbname=self.config["pg_database"],
                user=self.config["pg_user"],
                password=self.config["pg_password"],
                host=self.config["pg_host"],
                port=self.config["pg_port"]
            )
            self._init_postgres_table()

    def _init_postgres_table(self):
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(255) NOT NULL,
                    query_text TEXT NOT NULL,
                    query_type VARCHAR(50) NOT NULL,
                    parameters JSONB NOT NULL,
                    response JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    execution_time FLOAT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    error TEXT
                )
            """)
            self.pg_conn.commit()

    async def log_query(self, log_entry: QueryLog):
        if self.storage_type == "azure":
            await self._log_to_azure(log_entry)
        else:
            await self._log_to_postgres(log_entry)

    async def _log_to_azure(self, log_entry: QueryLog):
        date_path = log_entry.timestamp.strftime("%Y/%m/%d")
        file_name = f"{date_path}/{log_entry.timestamp.strftime('%H%M%S')}_{log_entry.model_id}.json"
        
        file_client = self.container_client.get_file_client(file_name)
        content = log_entry.json()
        
        await asyncio.to_thread(
            file_client.upload_data,
            data=content,
            overwrite=True
        )

    async def _log_to_postgres(self, log_entry: QueryLog):
        query = """
            INSERT INTO query_logs (
                model_id, query_text, query_type, parameters,
                response, timestamp, execution_time, status, error
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            log_entry.model_id,
            log_entry.query_text,
            log_entry.query_type,
            Json(log_entry.parameters),
            Json(log_entry.response),
            log_entry.timestamp,
            log_entry.execution_time,
            log_entry.status,
            log_entry.error
        )
        
        await asyncio.to_thread(
            self._execute_postgres_insert,
            query,
            values
        )

    def _execute_postgres_insert(self, query: str, values: tuple):
        with self.pg_conn.cursor() as cur:
            cur.execute(query, values)
            self.pg_conn.commit()

class SnowflakeConnector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conn = snowflake.connector.connect(
            user=config["snowflake_user"],
            password=config["snowflake_password"],
            account=config["snowflake_account"],
            warehouse=config["snowflake_warehouse"],
            database=config["snowflake_database"],
            schema=config["snowflake_schema"]
        )

    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            with self.conn.cursor() as cur:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                
                columns = [col[0] for col in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
                
                return {
                    "status": "success",
                    "data": results,
                    "row_count": len(results)
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 