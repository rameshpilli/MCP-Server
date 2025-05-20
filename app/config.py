import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    # Base configuration
    BASE_DIR = Path(__file__).parent.parent
    
    # Environment Detection
    IN_KUBERNETES: bool = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    
    # Server Configuration
    HOST: str = os.getenv("COHERE_MCP_SERVER_HOST", "localhost")
    PORT: int = int(os.getenv("COHERE_MCP_SERVER_PORT", "8000"))
    TRANSPORT: str = os.getenv("TRANSPORT", "sse")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    
    # CRM Server Branding
    SERVER_NAME: str = os.getenv("SERVER_NAME", "CRM MCP Server")
    SERVER_DESCRIPTION: str = os.getenv("SERVER_DESCRIPTION", "MCP Server for CRM information and financial tools")

    # LLM Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-opus-20240229")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1/messages")
    
    # OAuth Configuration for LLM
    LLM_OAUTH_ENDPOINT = os.getenv("LLM_OAUTH_ENDPOINT", "")
    LLM_OAUTH_CLIENT_ID = os.getenv("LLM_OAUTH_CLIENT_ID", "")
    LLM_OAUTH_CLIENT_SECRET = os.getenv("LLM_OAUTH_CLIENT_SECRET", "")
    LLM_OAUTH_GRANT_TYPE = os.getenv("LLM_OAUTH_GRANT_TYPE", "client_credentials")
    LLM_OAUTH_SCOPE = os.getenv("LLM_OAUTH_SCOPE", "")
    
    # Cohere Configuration
    COHERE_INDEX_NAME = os.getenv("COHERE_INDEX_NAME", "mcp_index")
    COHERE_SERVER_URL = os.getenv("COHERE_SERVER_URL", "")
    COHERE_SERVER_BEARER_TOKEN = os.getenv("COHERE_SERVER_BEARER_TOKEN", "")
    
    # MCP Server Configuration
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8080"))
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
    
    # Logging Configuration
    LOG_DIR = BASE_DIR / "logs"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Document Storage
    DOCS_DIR = BASE_DIR / "docs"

    # Redis Configuration (for session memory)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # API Configuration
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")

    # Cohere Compass Configuration
    COMPASS_API_URL = os.getenv("COMPASS_API_URL", "")
    COMPASS_BEARER_TOKEN = os.getenv("COMPASS_BEARER_TOKEN", "")
    COMPASS_INDEX_NAME = os.getenv("COMPASS_INDEX_NAME", "mcp_routing")

    # Financial Data Configuration
    FINANCIAL_API_BASE_URL = os.getenv("FINANCIAL_API_BASE_URL", "http://localhost:8002")
    SUPPORTED_CURRENCIES = os.getenv("SUPPORTED_CURRENCIES", "USD,CAD").split(",")

    @classmethod
    def get_api_url(cls) -> str:
        """Get the full API URL"""
        return f"http://{cls.HOST}:{cls.PORT}{cls.API_PREFIX}"

    @classmethod
    def get_redis_url(cls) -> str:
        """Get the Redis URL"""
        auth = f":{cls.REDIS_PASSWORD}@" if cls.REDIS_PASSWORD else ""
        return f"redis://{auth}{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"

    @classmethod
    def get_safe_config(cls) -> dict:
        """Get configuration without sensitive information"""
        safe_config = {
            "host": cls.HOST,
            "port": cls.PORT,
            "transport": cls.TRANSPORT,
            "debug": cls.DEBUG,
            "server_name": cls.SERVER_NAME,
            "server_description": cls.SERVER_DESCRIPTION,
            "cohere_index_name": cls.COHERE_INDEX_NAME,
            "llm_model": cls.LLM_MODEL,
            "llm_oauth_scope": cls.LLM_OAUTH_SCOPE,
            "api_prefix": cls.API_PREFIX,
            "cors_origins": cls.CORS_ORIGINS,
            "compass_api_url": cls.COMPASS_API_URL,
            "compass_index_name": cls.COMPASS_INDEX_NAME
        }
        return safe_config

# Create global config instance
config = Config() 