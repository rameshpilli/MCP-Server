# Model Context Protocol (MCP) Integration

This document provides technical details about the Model Context Protocol (MCP) integration in this project, including its architecture, components, and how it's implemented.

## Overview

The Model Context Protocol (MCP) is a standardized interface for interacting with language models and data sources. Our implementation provides:

1. A custom model registry for managing AI models
2. Standardized data source connections (Snowflake, Azure Storage, S3)
3. An MCP-compatible server that can be used with Claude and other MCP clients
4. Integration with our existing application architecture

## Architecture

The MCP integration consists of the following components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI App   │─────│   MCP Server    │─────│   MCP Database  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                        │
         │                      │                        │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Web UI/Clients │     │ Storage Backends│     │  Model Clients  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **FastAPI App**: Main application that integrates the MCP server
- **MCP Server**: Provides the Model Context Protocol interface
- **MCP Database**: Stores model and data source configurations
- **Storage Backends**: Handles data access to various sources
- **Model Clients**: Manages connections to AI model providers

## Core Components

### 1. Database Module (`mcp/database/`)

The database module provides:
- SQLAlchemy models for data structures (models, data sources, API keys)
- Connection management with async support
- CRUD operations for models and data sources

Key files:
- `mcp/database/__init__.py`: Database connection management
- `mcp/database/models.py`: SQLAlchemy models and CRUD operations
- `mcp/database/config.py`: Database configuration

### 2. Storage Module (`mcp/storage/`)

The storage module provides a unified interface for different storage backends:
- `SnowflakeStorage`: For Snowflake data warehouse connections
- `S3Storage`: For AWS S3 integration
- `AzureStorage`: For Azure Blob Storage
- `LocalStorage`: For local filesystem access

Each storage backend implements a common interface with methods like:
- `read_file(path)`: Read data from a path
- `write_file(path, data)`: Write data to a path
- `list_files(path)`: List files in a directory
- `delete_file(path)`: Delete a file

### 3. MCP Server (`app/api/mcp_server.py`)

The MCP server exposes resources and tools through the Model Context Protocol:

**Resources:**
- `/models`: List all registered models
- `/models/{model_id}`: Get details for a specific model
- `/sources`: List all registered data sources
- `/snowflake/{source}/{path}`: Access data from Snowflake
- `/azure/{source}/{path}`: Access data from Azure Storage
- `/s3/{source}/{path}`: Access data from S3

**Tools:**
- `query_snowflake`: Execute SQL queries against Snowflake
- `generate_with_model`: Generate text using a specific model

### 4. Standalone Server (`mcp_server.py`)

The standalone server provides a way to run the MCP server directly:
- Can be used with the MCP CLI
- Compatible with Claude Desktop and other MCP clients
- Manages database connections and server lifecycle

## Data Models

### Model

```python
class Model(Base):
    __tablename__ = "mcp_models"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    version = Column(String, nullable=True)
    api_base = Column(String, nullable=True)
    backend = Column(String, nullable=False)
    configuration = Column(JSON, nullable=True, default={})
    is_active = Column(Boolean, default=True)
    metrics = Column(JSON, nullable=True, default={})
```

### DataSource

```python
class DataSource(Base):
    __tablename__ = "mcp_data_sources"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    type = Column(String, nullable=False)
    description = Column(String, nullable=True)
    configuration = Column(JSON, nullable=True, default={})
    is_active = Column(Boolean, default=True)
```

## Integration Flow

1. When the application starts, it initializes the MCP database
2. The FastAPI app mounts the MCP server as a sub-application at `/mcp`
3. Clients can interact with the MCP server through HTTP requests
4. The MCP server handles resource and tool requests, using the database and storage backends
5. For model generation, the MCP server uses the model client factory to create appropriate clients

## Using with Claude

Claude can be configured to use our MCP server through its system prompt:

```
You are Claude, an AI assistant augmented with MCP capabilities.
You have access to the following Model Context Protocol server: http://localhost:8000/mcp

The server exposes these resources:
- /models - List all available models
- /models/<model_id> - Get information about a specific model
- /sources - List all available data sources
- /snowflake/<source>/<path> - Access data from Snowflake
- /azure/<source>/<path> - Access data from Azure Storage
- /s3/<source>/<path> - Access data from S3

You can also use these tools:
- query_snowflake - Execute SQL queries against Snowflake
- generate_with_model - Generate text using a specific model
```

## Demo Scripts

The project includes demo scripts to showcase MCP functionality:

1. `demo_mcp.py`: Demonstrates registering models and data sources
2. `demo_claude_mcp.py`: Shows how to use Claude with our MCP implementation
3. `test_mcp_simple.py`: Tests basic MCP functionality
4. `test_mcp_integration.py`: Tests database and server integration

## Configuration

The MCP integration uses the following environment variables:

```
# Database configuration
MCP_DATABASE_URL=sqlite+aiosqlite:///./data/mcp.db
MCP_DATABASE_ECHO=false

# Storage configuration
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AZURE_CONNECTION_STRING=your-connection-string

# Claude configuration
CLAUDE_API_KEY=your-claude-api-key
```

## Next Steps and Future Improvements

1. **Enhanced Security**: Implement more robust authentication and permission models
2. **Additional Storage Backends**: Add support for more data sources like BigQuery, PostgreSQL, etc.
3. **Caching Layer**: Implement caching for frequent queries and model responses
4. **Monitoring**: Add comprehensive logging and monitoring for MCP server operations
5. **Interactive UI**: Develop a web UI for managing models and data sources 