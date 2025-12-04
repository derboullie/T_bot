"""Logging configuration and setup."""

import sys
from pathlib import Path
from loguru import logger

from backend.core.config import settings


def setup_logging():
    """
    Configure logging for the application.
    
    Sets up:
    - Console logging with color
    - File logging with rotation
    - Different log levels for dev/prod
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File handler with rotation (create logs directory if needed)
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        enqueue=True,  # Async logging for better performance
    )
    
    logger.info(f"Logging initialized - Level: {settings.log_level}")


# Initialize logging
setup_logging()
