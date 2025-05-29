#!/bin/bash
set -e

# MCP Server Docker Entry Point
# This script handles different run modes, environment validation,
# health checks, and graceful shutdown for the MCP server.

# Color codes for better visibility in logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Validate required environment variables
validate_environment() {
    local missing_vars=0
    
    # Check required variables
    if [ -z "$MCP_SERVER_HOST" ]; then
        log_warning "MCP_SERVER_HOST is not set, defaulting to 0.0.0.0"
        export MCP_SERVER_HOST="0.0.0.0"
    fi
    
    if [ -z "$MCP_SERVER_PORT" ]; then
        log_warning "MCP_SERVER_PORT is not set, defaulting to 8080"
        export MCP_SERVER_PORT="8080"
    fi
    
    # Optional variables with defaults
    if [ -z "$SERVER_NAME" ]; then
        log_warning "SERVER_NAME is not set, defaulting to 'MCP Server'"
        export SERVER_NAME="MCP Server"
    fi
    
    if [ -z "$SERVER_DESCRIPTION" ]; then
        log_warning "SERVER_DESCRIPTION is not set, defaulting to 'Model Context Protocol Server'"
        export SERVER_DESCRIPTION="Model Context Protocol Server"
    fi
    
    if [ -z "$LOG_LEVEL" ]; then
        log_warning "LOG_LEVEL is not set, defaulting to INFO"
        export LOG_LEVEL="INFO"
    fi
    
    if [ -z "$ENVIRONMENT" ]; then
        log_warning "ENVIRONMENT is not set, defaulting to production"
        export ENVIRONMENT="production"
    fi
    
    # Check if we need to use mock data
    if [ "$USE_MOCK_DATA" = "true" ]; then
        log_info "Mock data enabled, will start mock data server"
    fi
    
    # Return success if all required variables are set
    return $missing_vars
}

# Start the mock data server if needed
start_mock_server() {
    if [ "$USE_MOCK_DATA" = "true" ]; then
        log_info "Starting mock financial data server on port 8001..."
        python -m examples.dummy_financial_server &
        MOCK_PID=$!
        log_success "Mock server started with PID $MOCK_PID"
        
        # Wait for mock server to be ready
        sleep 2
        
        # Set environment variable to point to mock server
        export CLIENTVIEW_BASE_URL="http://localhost:8001"
    fi
}

# Start the MCP server in HTTP mode
start_http_mode() {
    log_info "Starting MCP server in HTTP mode on $MCP_SERVER_HOST:$MCP_SERVER_PORT"
    
    # Use streamlined server if available, fall back to main_factory
    if [ -f "app/streamlined_mcp_server.py" ]; then
        log_info "Using streamlined MCP server"
        exec uvicorn app.streamlined_mcp_server:app \
            --host "$MCP_SERVER_HOST" \
            --port "$MCP_SERVER_PORT" \
            --proxy-headers \
            --log-level "$LOG_LEVEL" \
            --timeout-keep-alive 75
    else
        log_info "Using standard MCP server with factory"
        exec uvicorn app.mcp_server:main_factory \
            --host "$MCP_SERVER_HOST" \
            --port "$MCP_SERVER_PORT" \
            --factory \
            --proxy-headers \
            --log-level "$LOG_LEVEL" \
            --timeout-keep-alive 75
    fi
}

# Start the MCP server in STDIO mode
start_stdio_mode() {
    log_info "Starting MCP server in STDIO mode"
    
    # Use streamlined server if available, fall back to original
    if [ -f "app/streamlined_mcp_server.py" ]; then
        log_info "Using streamlined MCP server"
        exec python -m app.streamlined_mcp_server --mode stdio
    else
        log_info "Using standard MCP server"
        exec python -m app.mcp_server --mode stdio
    fi
}

# Start Chainlit UI if enabled
start_ui() {
    if [ "$ENABLE_UI" = "true" ]; then
        log_info "Starting Chainlit UI on port $UI_PORT..."
        chainlit run ui/app.py --host "$MCP_SERVER_HOST" --port "$UI_PORT" &
        UI_PID=$!
        log_success "Chainlit UI started with PID $UI_PID"
    else
        log_info "UI disabled, skipping Chainlit startup"
    fi
}

# Perform a health check
health_check() {
    local max_attempts=30
    local attempt=1
    local wait_seconds=1
    
    log_info "Waiting for MCP server to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://$MCP_SERVER_HOST:$MCP_SERVER_PORT/ping" > /dev/null; then
            log_success "MCP server is ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - Server not ready yet, waiting $wait_seconds seconds..."
        sleep $wait_seconds
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Cleanup function for graceful shutdown
cleanup() {
    log_info "Received signal to shut down..."
    
    # Kill mock server if running
    if [ ! -z "$MOCK_PID" ]; then
        log_info "Stopping mock server (PID: $MOCK_PID)"
        kill -TERM "$MOCK_PID" 2>/dev/null || true
    fi
    
    # Kill UI if running
    if [ ! -z "$UI_PID" ]; then
        log_info "Stopping Chainlit UI (PID: $UI_PID)"
        kill -TERM "$UI_PID" 2>/dev/null || true
    fi
    
    log_success "Cleanup complete, exiting"
    exit 0
}

# Set up signal handling for graceful shutdown
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting MCP Server Docker entrypoint"
    
    # Validate environment variables
    validate_environment
    
    # Create logs directory if it doesn't exist
    mkdir -p /usr/src/elements-ai-server/logs
    
    # Set default values for optional parameters
    : ${SERVER_MODE:="http"}
    : ${ENABLE_UI:="false"}
    : ${UI_PORT:="8081"}
    : ${USE_MOCK_DATA:="false"}
    
    # Start mock server if needed
    start_mock_server
    
    # Start UI if enabled
    start_ui
    
    # Start MCP server based on mode
    case "$SERVER_MODE" in
        http|HTTP)
            start_http_mode
            ;;
        stdio|STDIO)
            start_stdio_mode
            ;;
        *)
            log_error "Unknown server mode: $SERVER_MODE"
            log_error "Valid modes are: http, stdio"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
