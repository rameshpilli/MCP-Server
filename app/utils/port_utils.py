import socket
import logging

logger = logging.getLogger(__name__)

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """
    Find an available port starting from start_port.
    
    Args:
        start_port (int): The port to start checking from
        max_attempts (int): Maximum number of ports to try after start_port
        
    Returns:
        int: An available port number, or None if no port is available
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            # Create a socket and try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                logger.info(f"Found available port: {port}")
                return port
        except OSError:
            # Port is in use, try the next one
            continue
    
    logger.error(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")
    return None 