from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from enum import Enum
from dataclasses import dataclass
from sqlalchemy import create_engine, Column, String, JSON, DateTime, ForeignKey, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class DataSourceMetadata(Base):
    __tablename__ = 'data_sources'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # snowflake, azure, local_db
    description = Column(String)
    config = Column(JSON)  # Connection details, schemas, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    tables = relationship("TableMetadata", back_populates="data_source")

class TableMetadata(Base):
    __tablename__ = 'tables'
    
    id = Column(String, primary_key=True)
    data_source_id = Column(String, ForeignKey('data_sources.id'))
    name = Column(String, nullable=False)
    schema = Column(String)
    description = Column(String)
    topic = Column(String)  # e.g., 'jobs', 'models', 'metrics'
    columns = Column(JSON)  # Column definitions
    sample_queries = Column(JSON)  # Example queries for this table
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    data_source = relationship("DataSourceMetadata", back_populates="tables")

class QueryHistory(Base):
    __tablename__ = 'query_history'
    
    id = Column(String, primary_key=True)
    api_key = Column(String, nullable=False)
    natural_query = Column(String)
    sql_query = Column(String)
    data_source_id = Column(String, ForeignKey('data_sources.id'))
    execution_time = Column(Float)
    row_count = Column(Integer)
    status = Column(String)
    error = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    response = Column(JSON)

class MetadataManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    async def register_data_source(self, source_data: Dict[str, Any]) -> DataSourceMetadata:
        """Register a new data source with its metadata"""
        session = self.Session()
        try:
            data_source = DataSourceMetadata(
                id=source_data["id"],
                name=source_data["name"],
                type=source_data["type"],
                description=source_data.get("description"),
                config=source_data.get("config", {})
            )
            session.add(data_source)
            session.commit()
            return data_source
        finally:
            session.close()

    async def register_table(self, table_data: Dict[str, Any]) -> TableMetadata:
        """Register a new table with its metadata"""
        session = self.Session()
        try:
            table = TableMetadata(
                id=table_data["id"],
                data_source_id=table_data["data_source_id"],
                name=table_data["name"],
                schema=table_data.get("schema"),
                description=table_data.get("description"),
                topic=table_data.get("topic"),
                columns=table_data.get("columns", {}),
                sample_queries=table_data.get("sample_queries", [])
            )
            session.add(table)
            session.commit()
            return table
        finally:
            session.close()

    async def get_table_by_topic(self, topic: str) -> List[TableMetadata]:
        """Get all tables related to a specific topic"""
        session = self.Session()
        try:
            return session.query(TableMetadata).filter(TableMetadata.topic == topic).all()
        finally:
            session.close()

    async def log_query(self, query_data: Dict[str, Any]) -> QueryHistory:
        """Log a query execution"""
        session = self.Session()
        try:
            query_log = QueryHistory(
                id=query_data["id"],
                api_key=query_data["api_key"],
                natural_query=query_data.get("natural_query"),
                sql_query=query_data.get("sql_query"),
                data_source_id=query_data["data_source_id"],
                execution_time=query_data.get("execution_time"),
                row_count=query_data.get("row_count"),
                status=query_data["status"],
                error=query_data.get("error"),
                response=query_data.get("response")
            )
            session.add(query_log)
            session.commit()
            return query_log
        finally:
            session.close()

    async def get_data_source_config(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a data source"""
        session = self.Session()
        try:
            data_source = session.query(DataSourceMetadata).filter(
                DataSourceMetadata.id == source_id,
                DataSourceMetadata.is_active == True
            ).first()
            return data_source.config if data_source else None
        finally:
            session.close()

    async def get_table_metadata(self, table_id: str) -> Optional[TableMetadata]:
        """Get metadata for a specific table"""
        session = self.Session()
        try:
            return session.query(TableMetadata).filter(TableMetadata.id == table_id).first()
        finally:
            session.close()

    async def get_query_history(self, api_key: str, limit: int = 100) -> List[QueryHistory]:
        """Get query history for an API key"""
        session = self.Session()
        try:
            return session.query(QueryHistory)\
                .filter(QueryHistory.api_key == api_key)\
                .order_by(QueryHistory.created_at.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close() 