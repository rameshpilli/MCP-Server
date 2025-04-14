import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server Configuration
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000")

# Authentication
API_KEY = os.getenv("MCP_API_KEY")  # Load from environment variable

# If no API key in environment, use this for testing (NOT FOR PRODUCTION)
if not API_KEY:
    API_KEY = None  # Will be obtained through API call

# Model Configuration
MODEL_NAME = "test-llm"
MODEL_VERSION = "0.1"
MODEL_DESCRIPTION = "Test LLM for MCP integration" 