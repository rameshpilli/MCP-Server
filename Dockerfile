# Use official slim image for a smaller production footprint
FROM python:3.11-slim
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


# Set up application directory
WORKDIR /usr/src/elements-ai-server

# Copy application code
COPY app/ ./app/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY ui/ ./ui/
COPY run.py .
COPY mcp_client.py .
COPY client_cli.py .
COPY pyproject.toml .
# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Set environment variables for container
ENV IN_KUBERNETES="true"
ENV LOG_TO_STDOUT="true"
ENV COHERE_MCP_SERVER_HOST="0.0.0.0"
ENV COHERE_MCP_SERVER_PORT="8000"
ENV MCP_SERVER_HOST="localhost"
ENV MCP_SERVER_PORT="8081"

# Expose ports for API, MCP server, and UI
EXPOSE 8000 8081 8501

# Create and set permissions for logs directory
RUN mkdir -p /usr/src/elements-ai-server/logs && \
    chmod -R 755 /usr/src/elements-ai-server/logs

VOLUME ["/usr/src/elements-ai-server/logs"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start application
CMD ["python", "run.py"]
