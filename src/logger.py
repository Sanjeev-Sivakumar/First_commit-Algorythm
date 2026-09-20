"""
KineticGuard Logging Module
Provides structured, thread-safe, and colorized logging for multi-camera video ingestion.
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

console = Console()


class KineticGuardFormatter(logging.Formatter):
    """Custom formatter with clear alignment and event tagging."""

    def format(self, record: logging.LogRecord) -> str:
        camera_id = getattr(record, "camera_id", "SYSTEM")
        event = getattr(record, "event", "INFO")
        record.camera_tag = f"[{camera_id}]"
        record.event_tag = f"[{event}]"
        return super().format(record)


def setup_logger(name: str = "KineticGuard", level: str = "INFO") -> logging.Logger:
    """Configures and returns the main application logger."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Rich console handler for beautiful colored outputs
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = setup_logger()


def log_event(
    camera_id: str,
    event: str,
    message: str,
    level: str = "info",
    extra_details: Optional[dict] = None,
):
    """
    Log an event with camera ID and standardized event categorization.

    Event types:
        STREAM_INIT, CONNECTED, INGEST, ACK, RECONNECT, WARNING, ERROR, COMPLETED
    """
    color_map = {
        "STREAM_INIT": "cyan",
        "CONNECTED": "bold green",
        "INGEST": "blue",
        "ACK": "green",
        "RECONNECT": "bold yellow",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "COMPLETED": "magenta",
    }
    color = color_map.get(event, "white")
    tag = f"[{color}][{event}][/{color}]"
    cam_str = f"[bold white]({camera_id})[/bold white]" if camera_id else ""

    formatted_msg = f"{cam_str:<22} {tag:<18} {message}"
    if extra_details:
        formatted_msg += f" [dim]({extra_details})[/dim]"

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(formatted_msg, extra={"camera_id": camera_id, "event": event})
