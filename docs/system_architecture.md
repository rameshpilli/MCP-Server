# System Architecture Overview

## Components

### 1. Model Context Protocol (MCP)
- Handles all model interactions
- Manages context and tool execution
- Provides OAuth authentication for model access

### 2. Cohere Compass Integration
- Semantic search capabilities
- Document indexing and retrieval
- Real-time context enhancement

### 3. Local Document Store
- Location: `/docs` directory
- Supports markdown and text files
- Used for storing internal documentation
- Accessible via document search tool

## Configuration

### Environment Variables
- `LLM_MODEL`: Model identifier
- `COHERE_INDEX_NAME`: Index for semantic search
- `LLM_BASE_URL`: Base URL for model endpoint

### Security
- OAuth2 authentication for model access
- Bearer token for Cohere API
- Local file system security

## Deployment
- Supports both local and Kubernetes deployment
- Configuration via environment variables
- Logging to both CSV and file outputs 