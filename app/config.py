# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Load environment variables - for local development
load_dotenv()

class Config:
    # Base configuration
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Server Version and Environment
    VERSION: str = os.getenv("SERVER_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Path to parameter mappings file
    PARAMETER_MAPPINGS_PATH: str = os.getenv(
        "PARAMETER_MAPPINGS_PATH",
        str(BASE_DIR / "config" / "parameter_mappings.json"),
    )

    # Server Configuration - Now with proper Kubernetes-friendly settings
    # We bind to 0.0.0.0 in k8s environments, but localhost for local dev
    HOST: str = os.getenv("COHERE_MCP_SERVER_HOST", "0.0.0.0" if os.getenv("KUBERNETES_SERVICE_HOST") else "localhost")
    PORT: int = int(os.getenv("COHERE_MCP_SERVER_PORT", "8001"))
    TRANSPORT: str = os.getenv("TRANSPORT", "sse")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")

    # CRM Server Branding
    SERVER_NAME: str = os.getenv("SERVER_NAME", "CRM MCP SERVER")
    SERVER_DESCRIPTION: str = os.getenv("SERVER_DESCRIPTION", "MCP Server for CRM information and financial tools")

    # Cohere Configuration
    COHERE_INDEX_NAME: Optional[str] = os.getenv("COHERE_INDEX_NAME")
    COHERE_SERVER_URL: Optional[str] = os.getenv("COHERE_SERVER_URL")
    COHERE_SERVER_BEARER_TOKEN: Optional[str] = os.getenv("COHERE_SERVER_BEARER_TOKEN")
    COHERE_PARSER_URL: Optional[str] = os.getenv("COHERE_PARSER_URL")
    COHERE_PARSER_BEARER_TOKEN: Optional[str] = os.getenv("COHERE_PARSER_BEARER_TOKEN")
    COHERE_SSL_VERIFY: bool = os.getenv("COHERE_SSL_VERIFY", "true").lower() != "false"

    # Cohere Compass Configuration (using the same values)
    COMPASS_API_URL: Optional[str] = os.getenv("COHERE_SERVER_URL")
    COMPASS_BEARER_TOKEN: Optional[str] = os.getenv("COHERE_SERVER_BEARER_TOKEN")
    COMPASS_INDEX_NAME: Optional[str] = os.getenv("COHERE_INDEX_NAME")

    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_OPENAI_MODEL", "gpt-4o-2024-05-13")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_OPENAI_BASE_URL")

    # OAuth Configuration for LLM
    LLM_OAUTH_ENDPOINT: Optional[str] = os.getenv("LLM_OPENAI_OAUTH_ENDPOINT")
    LLM_OAUTH_CLIENT_ID: Optional[str] = os.getenv("LLM_OPENAI_OAUTH_CLIENT_ID")
    LLM_OAUTH_CLIENT_SECRET: Optional[str] = os.getenv("LLM_OPENAI_OAUTH_CLIENT_SECRET")
    LLM_OAUTH_GRANT_TYPE: Optional[str] = os.getenv("LLM_OPENAI_OAUTH_GRANT_TYPE")
    LLM_OAUTH_SCOPE: str = os.getenv("LLM_OPENAI_OAUTH_SCOPE", "read")
    LLM_SUPPORTS_TEMPERATURE: bool = os.getenv("LLM_OPENAI_SUPPORTS_TEMPERATURE", "false").lower() == "true"

    # MCP Server Configuration - Now with Kubernetes service discovery
    # If we're in k8s, we use the service name, otherwise localhost
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", 
                                     "mcp-server-service" if os.getenv("KUBERNETES_SERVICE_HOST") else "localhost")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8081"))
    # Dynamic URL construction for service discovery
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}")

    # Logging Configuration - Modified for Kubernetes
    # In k8s, we'll use an ephemeral volume mount or object storage
    LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    # Option to log to stdout for container environments
    LOG_TO_STDOUT: bool = os.getenv("LOG_TO_STDOUT", "true").lower() == "true"

    # Document Storage - Allow external storage paths
    DOCS_DIR = os.getenv("DOCS_DIR", str(BASE_DIR / "docs"))

    # Redis Configuration (for session memory)
    # In k8s, Redis would likely be a separate service
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis-service" if os.getenv("KUBERNETES_SERVICE_HOST") else "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # API Configuration
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")

    # S3 Configuration for persistent storage
    S3_SERVER: str = os.getenv("RBC_S3_SERVER")
    S3_BUCKET_NAME: str = os.getenv("RBC_S3_BUCKET_NAME")
    S3_OUTPUT_FOLDER: str = os.getenv("RBC_S3_OUTPUT_FOLDER", "")
    S3_USER: str = os.getenv("RBC_S3_USER")
    S3_PASS: str = os.getenv("RBC_S3_PASS")
    S3_VERIFY_SSL: bool = os.getenv("RBC_S3_VERIFY_SSL", "false").lower() == "true"

    # Optional caching and vector usage
    USE_SQLITE_CACHE: bool = os.getenv("USE_SQLITE_CACHE", "true").lower() == "true"
    USE_VECTOR_DB: bool = os.getenv("USE_VECTOR_DB", "false").lower() == "true"
    PLAN_LOG_FILE: str = os.getenv("PLAN_LOG_FILE", str(BASE_DIR / "output" / "plan_debug.json"))
    
    # Kubernetes specific configuration
    IN_KUBERNETES: bool = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    POD_NAME: str = os.getenv("POD_NAME", "unknown-pod")
    NAMESPACE: str = os.getenv("POD_NAMESPACE", "default")

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
        return {
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
            "in_kubernetes": cls.IN_KUBERNETES,
            "pod_name": cls.POD_NAME,
            "namespace": cls.NAMESPACE
        }

    @classmethod
    def is_production(cls) -> bool:
        """Check if we're running in a production environment"""
        # We consider it production if we're in Kubernetes or ENV is set to prod
        return cls.IN_KUBERNETES or os.getenv("ENVIRONMENT", "").lower() == "production"

# Global config instance
config = Config()
