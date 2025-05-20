FROM innersource-docker/container-hub/python:3.11-slim
USER root

# Install Node.js and npm for MCP Inspector
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -sL http://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

# Install MCP Inspector globally
RUN npm install -g @modelcontextprotocol/inspector

# Copy certificates

# Configure pip to use Company's artifact repository
COPY artifacts/pip.conf /pip.conf
ENV PIP_CONFIG_FILE /pip.conf

# Set certificate environment variables

# Update certificate trust stores
RUN if command -v update-ca-certificates > /dev/null; then \
        update-ca-certificates; \
    elif command -v update-ca-trust > /dev/null; then \
        update-ca-trust; \
    else \
        echo "Warning: Neither update-ca-certificates nor update-ca-trust is available"; \
    fi

# Set up application directory
WORKDIR /usr/src/elements-ai-server

# Copy application code
COPY app/ ./app/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY ui/ ./ui/
COPY run.py .
COPY requirements.txt .
COPY artifacts/ ./artifacts/

# Install dependencies
RUN pip install --upgrade pip --timeout 60

# Install Companies Security directly from the wheel file


# Install compass_sdk from local directory if present
RUN if [ -d "./artifacts/compass-sdk" ]; then \
        pip install --no-index --find-links=./artifacts/compass-sdk compass_sdk || echo "Failed to install compass_sdk, continuing..."; \
    fi

# Install packages in batches with retry logic
# Core functionality
RUN pip install fastapi uvicorn httpx python-dotenv pydantic --default-timeout=60 || echo "Some core packages failed to install"

# MCP/LLM components
RUN pip install fastmcp anthropic openai cohere --default-timeout=60 || echo "Some MCP/LLM packages failed to install"

# Cloud storage and document handling
RUN pip install boto3 markdown beautifulsoup4 PyPDF2 "pdfminer.six" --default-timeout=60 || echo "Some document handling packages failed to install"

# Security and logging
RUN pip install python-multipart aiohttp --default-timeout=60 || echo "Some security packages failed to install"

# UI (if available)
RUN pip install chainlit --default-timeout=60 || echo "UI package failed to install"

# Additional packages
RUN pip install asyncio aiofiles pytest gunicorn prometheus-client tenacity tabulate --default-timeout=60 || echo "Some additional packages failed to install"

# Set environment variables for container
ENV IN_KUBERNETES="true"
ENV LOG_TO_STDOUT="true"
ENV COHERE_MCP_SERVER_HOST="0.0.0.0"
ENV COHERE_MCP_SERVER_PORT="8001"
ENV MCP_SERVER_HOST="localhost"
ENV MCP_SERVER_PORT="8081"

# Expose ports for API, MCP server, and UI
EXPOSE 8000 8001 8501

# Create and set permissions for logs directory
RUN mkdir -p /usr/src/elements-ai-server/logs && \
    chmod -R 755 /usr/src/elements-ai-server/logs

VOLUME ["/usr/src/elements-ai-server/logs"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start application
CMD ["python", "run.py"]
