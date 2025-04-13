# Local Development Setup Guide

This guide will help you set up the Model Context Protocol (MCP) on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- virtualenv or venv (recommended)

## Quick Setup

1. **Run the Setup Script**
   ```bash
   # On macOS/Linux
   ./scripts/setup.sh

   # On Windows
   scripts\setup.bat
   ```

   This will:
   - Create a virtual environment
   - Install all dependencies
   - Set up necessary directories
   - Create a `.env` file with secure defaults
   - Run tests to verify the installation

## Manual Setup (Alternative)

If you prefer to set up manually or the setup script doesn't work for you:

1. **Download and Extract**
   ```bash
   # If using git
   git clone https://github.com/yourusername/MCP.git
   cd MCP

   # If using zip file
   unzip MCP.zip
   cd MCP
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the Package**
   ```bash
   # Install in development mode with test dependencies
   pip install -e ".[test]"
   ```

4. **Configure Environment**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Generate a secure secret key
   python -c "import secrets; print(f'SECRET_KEY={secrets.token_hex(32)}')"
   ```
   Then edit `.env` and replace `your-secret-key-here` with the generated key.

## Environment Variables

The following environment variables can be configured in `.env`:

### Server Configuration
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Database Configuration
- `DATABASE_URL`: SQLite database URL
- `ASYNC_DATABASE_URL`: Async database URL (same as DATABASE_URL for SQLite)

### Security
- `SECRET_KEY`: Secret key for JWT encoding (generate unique)
- `API_KEY_EXPIRY_DAYS`: Days until API keys expire (default: 30)
- `ALGORITHM`: JWT algorithm (default: HS256)

### Logging
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_FILE`: Log file path (default: logs/mcp_server.log)

### Rate Limiting
- `RATE_LIMIT`: API rate limit (default: 20/minute)
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds (default: 60)

### Cache
- `CACHE_TTL`: Cache time-to-live in seconds (default: 300)

### File Storage
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 100MB)
- `UPLOAD_CHUNK_SIZE`: Upload chunk size in bytes (default: 1MB)
- `STORAGE_PATH`: Path to storage directory (default: storage)

### Model Configuration
- `DEFAULT_MODEL_TIMEOUT`: Default model timeout in seconds (default: 30)
- `MODEL_REGISTRY_PATH`: Path to model registry file (default: data/model_registry.json)

## Directory Structure

After installation, your project structure should look like this:
```
MCP/
├── app/                  # Main application code
├── data/                # Database and other data files
├── logs/                # Application logs
├── static/              # Static files
│   ├── css/
│   ├── js/
│   └── img/
├── storage/             # File storage
├── templates/           # HTML templates
└── tests/               # Test files
```

## Running the Application

1. **Start the Server**
   ```bash
   # Using uvicorn directly
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

   # Or using the installed console script
   mcp
   ```

2. **Access the API**
   - API Documentation: `http://localhost:8000/docs`
   - Health Check: `