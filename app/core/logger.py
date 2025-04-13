import logging
import logging.config
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure structlog pre-processors
pre_chain = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.stdlib.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": jsonlogger.JsonFormatter,
            "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "colored": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=True),
            "foreign_pre_chain": pre_chain,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "stream": sys.stdout,
        },
        "json_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "logs/app.json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "json_file"],
            "level": "INFO",
        },
        "app": {
            "handlers": ["console", "json_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "sqlalchemy.engine": {
            "handlers": ["console", "json_file"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["console", "json_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = structlog.get_logger("app")

def log_connection_info(host: str, port: int, mode: str) -> None:
    """Log server connection information."""
    logger.info(
        "server_startup",
        host=host,
        port=port,
        mode=mode,
        timestamp=datetime.utcnow().isoformat(),
        python_version=sys.version,
    )

def log_test_result(test_name: str, result: Dict[str, Any]) -> None:
    """Log test result information."""
    logger.info(
        "test_result",
        test_name=test_name,
        status=result.get("status"),
        duration=result.get("duration"),
        timestamp=datetime.utcnow().isoformat(),
    ) 