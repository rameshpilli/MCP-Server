# Model Context Protocol (MCP)

A FastAPI-based application for managing ML model contexts and deployments with built-in API key management, rate limiting, and PostgreSQL integration.

## Table of Contents
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL 12+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run the application:
```bash
python run.py
```

## Project Structure

```
mcp/
├── app/
│   ├── api/              # API endpoints
│   │   ├── auth.py      # Authentication
│   │   ├── config.py    # Settings
│   │   ├── database.py  # Database
│   │   ├── middleware.py # Middleware
│   │   └── security.py  # Security
│   └── models/          # Database models
├── tests/               # Test suite
├── main.py             # Application entry
├── run.py              # Server runner
└── requirements.txt    # Dependencies
```

## Core Components

### 1. Database Management (`app/core/database.py`)

Handles database connections and session management.

Key functions:
- `init_db()`: Initializes database and creates tables
- `get_db()`: Provides database session
- `create_database_if_not_exists()`: Auto-creates database

Example:
```python
# Get database session
async with get_db() as db:
    result = await db.execute(query)
    await db.commit()
```

### 2. Configuration (`app/core/config.py`)

Manages application settings using Pydantic.

Key settings:
- Server configuration (host, port)
- Database credentials
- API settings
- Security settings

Example:
```python
from app.core.config import get_settings

settings = get_settings()
db_url = settings.get_db_url()
```

### 3. Authentication (`app/core/auth.py`)

Handles API key management and validation.

Key classes:
- `APIKeyManager`: Generates and validates API keys
- Methods:
  - `generate_key()`: Creates new API key
  - `validate_key()`: Validates existing key

Example:
```python
api_key_manager = APIKeyManager()
key, api_key = await api_key_manager.generate_key(
    owner="user@example.com",
    permissions=["read", "write"]
)
```

### 4. Middleware (`app/core/middleware.py`)

Contains request processing middleware.

Components:
- `APIKeyMiddleware`: Validates API keys
- `UsageTrackingMiddleware`: Tracks API usage
- `RateLimiter`: Handles rate limiting

Example:
```python
app.add_middleware(APIKeyMiddleware)
app.add_middleware(UsageTrackingMiddleware)
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns application health status.

### API Keys
```http
POST /api/keys
GET /api/keys
DELETE /api/keys/{key_id}
```

### Models
```http
GET /api/models
POST /api/models
GET /api/models/{model_id}
```

## Configuration

Environment variables (can be set in `.env`):

```env
# Server
APP_PORT=8000
APP_HOST=0.0.0.0
APP_DEBUG=false

# Database
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mcp

# Security
SECRET_KEY=your_secret_key
```

## Development

### Running in Debug Mode
```bash
python run.py --debug --reload
```

### Command Line Options
```bash
python run.py --help
```
Available options:
- `--host`: Host to bind to
- `--port`: Port number
- `--reload`: Enable auto-reload
- `--debug`: Enable debug mode
- `--workers`: Number of worker processes

## Testing

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app tests/
```

## Error Handling

The application includes comprehensive error handling:

1. Database Errors:
   - Connection failures
   - Migration errors
   - Transaction errors

2. API Errors:
   - Invalid API keys
   - Rate limit exceeded
   - Invalid requests

3. System Errors:
   - Port already in use
   - File system errors
   - Permission issues

Example error response:
```json
{
  "detail": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-03-14T12:00:00Z"
}
```

## Logging

Structured logging is implemented throughout:

```python
logger.info("API request received", 
    endpoint="/api/models",
    method="POST",
    user_id="123"
)
```

Logs are written to:
- Console (for development)
- File (configurable path)
- JSON format (for production)

## Production Deployment

For production:

1. Set environment variables:
```env
PRODUCTION=true
DEBUG=false
```

2. Use proper database credentials:
```env
DB_USER=production_user
DB_PASSWORD=secure_password
```

3. Run with multiple workers:
```bash
python run.py --workers 4
```

## Security Best Practices

1. API Key Management:
   - Keys are hashed before storage
   - Automatic expiration
   - Rate limiting per key

2. Database Security:
   - Parameterized queries
   - Connection pooling
   - Transaction management

3. Request Security:
   - CORS protection
   - Input validation
   - Rate limiting

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## License

[License Type] - See LICENSE file for details