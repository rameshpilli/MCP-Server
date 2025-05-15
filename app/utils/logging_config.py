"""
Logging configuration for the MCP server.

This module sets up logging to both file and console with proper formatting and rotation.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

def setup_logging(
    name: str = "mcp_server",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        log_level: Logging level
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and configure file handler
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logging()

# Export commonly used logging functions
def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Optional name for the logger. If None, returns the default logger.
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    return logger 