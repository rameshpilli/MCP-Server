import logging
import os
from pathlib import Path
from app.config import config


def setup_logging(name: str) -> logging.Logger:
    """Create or retrieve a logger with standardized configuration."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", config.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if not os.getenv("LOG_TO_STDOUT_ONLY"):
        Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(config.LOG_DIR) / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
