import os
import socket
import sys
import uvicorn
from app.core.config import get_settings
from app.core.logger import logger

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def main():
    """Run the application."""
    settings = get_settings()
    
    # Try to use configured port, or find an available one
    try:
        port = find_available_port(settings.PORT)
        if port != settings.PORT:
            logger.warning(f"Port {settings.PORT} is in use, using port {port} instead")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # Configure uvicorn
    config = uvicorn.Config(
        "main:app",
        host=settings.HOST,
        port=port,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level="info" if settings.DEBUG else "error"
    )

    # Start server
    try:
        server = uvicorn.Server(config)
        logger.info(f"Starting server at http://{settings.HOST}:{port}")
        server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 