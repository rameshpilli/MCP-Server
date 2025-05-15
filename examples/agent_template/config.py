"""
Agent Configuration

This module contains the configuration for the agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MCP Server connection
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# Agent information
AGENT_NAME = os.getenv("AGENT_NAME", "SampleAgent")
AGENT_DESCRIPTION = os.getenv("AGENT_DESCRIPTION", "Sample agent for demonstration purposes")
AGENT_NAMESPACE = os.getenv("AGENT_NAMESPACE", "sample_agent")
AGENT_CAPABILITIES = os.getenv("AGENT_CAPABILITIES", "search,processing,analysis").split(",")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0.0")
AGENT_AUTHOR = os.getenv("AGENT_AUTHOR", "Your Name")
AGENT_CONTACT = os.getenv("AGENT_CONTACT", "your.email@example.com")

# API Keys and credentials
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# Data storage
DATA_DIR = os.getenv("DATA_DIR", str(Path(__file__).parent / "data"))
CACHE_DIR = os.getenv("CACHE_DIR", str(Path(__file__).parent / "cache"))

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 