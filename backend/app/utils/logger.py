"""Sets up logging"""
import logging
from logging.config import dictConfig


_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

# one-time flag so tests don’t re-configure
_INITIALIZED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger once and only once."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    cfg = dict(_CONFIG)
    cfg["root"]["level"] = level
    dictConfig(cfg)
    _INITIALIZED = True