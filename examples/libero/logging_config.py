"""Centralized logging configuration for libero examples."""

import logging
from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the entire application.

    Args:
        level: The logging level to use (default: logging.INFO)

    This should be called once at the start of each main script.
    Safe to call multiple times (force=True ensures it overrides existing config).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
        ],
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: The name of the logger (typically __name__)

    Returns:
        A configured logger instance

    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name)
