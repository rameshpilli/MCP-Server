#!/bin/bash

# Make script executable
chmod +x start.sh

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p storage data

# Build and start the containers
echo "Building and starting MCP Server..."
docker-compose up --build -d

# Wait for the server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "MCP Server is ready!"
        echo "You can access the API at http://localhost:8000"
        echo "API Documentation is available at http://localhost:8000/docs"
        exit 0
    fi
    sleep 1
done

echo "Error: Server failed to start within 30 seconds"
docker-compose logs
exit 1 