# Model Context Protocol (MCP)

A platform for managing, monitoring, and optimizing model interactions with data sources.

## Features

- API Key authentication and management
- Configurable rate limiting
- Usage tracking and quotas
- Model registration and status monitoring
- Configurable storage backends

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Initialize the database:
```bash
alembic upgrade head
```

## Configuration

The platform can be configured using environment variables or a `.env` file. Key settings include:

### Database
- `POSTGRES_USER`: PostgreSQL username (default: postgres)
- `POSTGRES_PASSWORD`: PostgreSQL password (default: postgres)
- `POSTGRES_HOST`: Database host (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_DB`: Database name (default: mcp)

### Storage Backend
- `STORAGE_BACKEND`: Storage backend to use (local or azure)
- `AZURE_SQL_CONNECTION_STRING`: Azure SQL connection string (required for azure backend)

### API Keys
- `DEFAULT_API_KEY_EXPIRY_DAYS`: Default expiry for API keys (default: 30)
- `TEST_API_KEY`: Special test API key that bypasses rate limits
- `DEFAULT_RATE_LIMIT`: Default rate limit for API keys (default: 20/minute)

### Rate Limiting
- `RATE_LIMIT_CLEANUP_INTERVAL`: Interval for cleaning up old rate limit entries (default: 3600)
- `RATE_LIMIT_DEFAULT_LIMIT`: Default request limit (default: 20)
- `RATE_LIMIT_DEFAULT_WINDOW`: Default time window (minute or hour)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app

# Run specific test file
pytest tests/test_api.py
```

## Development

For local development:

1. Set up pre-commit hooks:
```bash
pre-commit install
```

2. Create a development database:
```bash
createdb mcp
alembic upgrade head
```

3. Run the development server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

[License information]