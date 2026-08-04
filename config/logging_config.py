"""
config/logging_config.py — Structured JSON Logging
==================================================
Day 16: Centralized structured logging for AgentLens.

Outputs to console and a rotating file handler (logs/agentlens.log)
kept for 7 days.
"""

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agentlens.log")


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge extra attributes if any
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        return json.dumps(log_obj)


def setup_logging():
    """Initialize the root logger configuration."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("agentlens")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicating handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = JSONFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Rotating file handler (daily rotation, keep 7 days)
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False
    return logger

# Initialize logging when module is imported
_root_logger = setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Get a child logger configured for JSON output."""
    return _root_logger.getChild(name)
