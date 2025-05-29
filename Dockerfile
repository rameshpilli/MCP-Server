# Use official slim image for a smaller production footprint
FROM python:3.11-slim

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js, npm, and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    build-essential \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /usr/src/mcp-server

# Copy entrypoint script first and make it executable
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir pandas>=2.0.0 && \
    pip install --no-cache-dir -r requirements-extra.txt

# Install npm dependencies
RUN npm install

# Create necessary directories
RUN mkdir -p logs output && \
    chmod -R 755 logs output

# Set environment variables
ENV IN_KUBERNETES="true" \
    LOG_TO_STDOUT="true" \
    MCP_SERVER_HOST="0.0.0.0" \
    MCP_SERVER_PORT="8080" \
    SERVER_MODE="http" \
    ENABLE_UI="false" \
    UI_PORT="8081" \
    USE_MOCK_DATA="false" \
    LOG_LEVEL="INFO" \
    ENVIRONMENT="production" \
    SERVER_NAME="MCP Server" \
    SERVER_DESCRIPTION="Model Context Protocol Server"

# Expose ports for API, UI, and mock data server
EXPOSE 8080 8081 8001

# Add healthcheck using the correct port
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1

# Volume for logs and output
VOLUME ["/usr/src/mcp-server/logs", "/usr/src/mcp-server/output"]

# Use entrypoint script to handle different run modes
ENTRYPOINT ["docker-entrypoint.sh"]
