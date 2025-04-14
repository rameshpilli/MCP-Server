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

4. Initialize the database:
   ```bash
   alembic upgrade head
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

## Development

### Project Structure

- `app/`: The main application
  - `api/`: API endpoints
    - `mcp_server.py`: MCP server implementation
  - `core/`: Core functionality
    - `storage.py`: Data source backends
  - `models/`: Database models
- `mcp_server.py`: Standalone MCP server entry point

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

MIT