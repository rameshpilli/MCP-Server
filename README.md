# Model Context Protocol (MCP) Server

A unified platform for managing models and data sources through the Model Context Protocol.

## Features

- **Custom Model Registry**: Register and manage multiple LLM models
- **Standardized MCP Interface**: Connect your models to applications using MCP
- **Data Source Integration**: Connect to Snowflake, Azure Storage, and S3
- **Web UI**: Manage models through a web interface
- **API Key Management**: Control access to your models

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MCP.git
   cd MCP
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start the server:
   ```bash
   # The database will be initialized automatically on first run
   uvicorn app.main:app --reload
   ```

## Usage

### Running the Web Server

Start the FastAPI web server:

```bash
uvicorn app.main:app --reload
```

Access the web UI at http://localhost:8000/llm/ui

### Running as an MCP Server

You can use this as a standalone MCP server for Claude Desktop or other MCP clients:

```bash
# Run directly
python mcp_server.py

# Or using the MCP CLI
mcp run mcp_server.py

# Install in Claude Desktop
mcp install mcp_server.py
```

## MCP Protocol Integration

This server implements the Model Context Protocol, providing:

### Resources

Access data through standardized resource URLs:

- `models://list` - List all registered models
- `models://{model_id}` - Details for a specific model
- `sources://list` - List data sources
- `snowflake://{source_name}/{path}` - Data from Snowflake
- `azure://{source_name}/{path}` - Data from Azure Storage
- `s3://{source_name}/{path}` - Data from S3

### Tools

Programmatically interact with models and data:

- `generate_with_model(model_id, prompt)` - Generate text using a model
- `query_snowflake(source_name, query)` - Run Snowflake queries
- `list_storage_files(source_name, path)` - List files in storage
- `register_model(model_id, name, backend)` - Register a new model

### Prompts

Use built-in prompt templates:

- `data_analysis_prompt(data, question)` - Create a data analysis prompt
- `query_generator_prompt(table_description, question)` - Generate SQL from natural language

## Configuring Data Sources

### Snowflake

Configure in the `.env` file:

```
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
```

### Azure Storage

Configure in the `.env` file:

```
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER_NAME=your_container
```

### S3

Configure in the `.env` file:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket
```

## Project Structure

- `app/`: The main application
  - `api/`: API endpoints
    - `mcp_server.py`: MCP server implementation
  - `core/`: Core functionality
    - `config.py`: Application configuration
    - `logger.py`: Logging setup
    - `model_client.py`: Model clients for different backends
- `mcp_server.py`: Standalone MCP server entry point
- `docs/`: Documentation
  - `architecture.md`: Detailed architecture explanation
  - `usage.md`: Usage guide

## Client SDK

For easy integration with your applications, use our Python client SDK:

```python
from mcp_client import MCPClient

# Initialize client
client = MCPClient(
    api_key="your_api_key", 
    base_url="http://localhost:8000/mcp"
)

# List models
models = client.list_models()

# Generate with a model
response = client.generate_with_model(
    model_id="gpt4-turbo",
    prompt="What's the weather like today?"
)
```

## License

MIT

# MCP (Model Context Protocol) Integration

This project integrates the Model Context Protocol (MCP) SDK to provide a standardized way of interacting with ML models and data sources.

## Features

- **Custom Model Registry**: Register and manage different types of models with a unified API
- **Standardized MCP Interface**: Use standard MCP resources to access models and data
- **Data Source Integration**: Connect to Snowflake, Azure Storage, and S3 data sources
- **Web UI**: Manage models and data sources through a web interface
- **API Key Management**: Secure access to models with API keys

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd MCP
   ```

2. Set up a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating a `.env` file:
   ```
   # Copy the example file
   cp .env.example .env
   
   # Edit with your settings
   # Important: Set DATABASE_URL to your database connection string
   ```

5. Initialize the database:
   ```
   python mcp_server.py initialize-db
   ```

## Usage

### Running the Web Server

To start the FastAPI web server:

```
uvicorn app.main:app --reload
```

This will start the server at http://127.0.0.1:8000.

### Running as MCP Server

To run as a standalone MCP server:

```
python mcp_server.py
```

### MCP Protocol Integration

The MCP server exposes the following resources:

- Models: `models://list`, `models://{model_id}`
- Data Sources: `sources://list`
- Snowflake Data: `snowflake://{source_name}/{path}`
- Azure Storage: `azure://{source_name}/{path}`
- S3 Data: `s3://{source_name}/{path}`

And tools:

- `query_snowflake`: Execute queries against Snowflake
- `generate_with_model`: Generate text with a specific model
- `list_storage_files`: List files in a storage path
- `register_model`: Register a new model

### Configuring Data Sources

#### Snowflake

Set the following environment variables or update your `.env` file:

```
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_WAREHOUSE=your-warehouse
SNOWFLAKE_DATABASE=your-database
SNOWFLAKE_SCHEMA=your-schema
```

#### Azure Storage

Set the following environment variables or update your `.env` file:

```
AZURE_STORAGE_ACCOUNT=your-storage-account
AZURE_STORAGE_KEY=your-storage-key
AZURE_CONTAINER_NAME=your-container-name
```

#### S3

Set the following environment variables or update your `.env` file:

```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=your-region
S3_BUCKET=your-bucket
```

## Development

### Project Structure

```
.
├── app/                # Main application code
│   ├── api/            # API endpoints
│   ├── core/           # Core functionality
│   └── models/         # Model definitions
├── mcp/                # MCP SDK integration
│   ├── database/       # Database functionality
│   ├── server/         # Server functionality
│   └── storage/        # Storage backends
├── storage/            # Storage directory for local files
├── mcp_server.py       # MCP server entry point
└── requirements.txt    # Dependencies
```

### Running Tests

```
python test_mcp_simple.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.