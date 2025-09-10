from __future__ import annotations

import logging

import structlog


def configure_logging() -> structlog.typing.WrappedLogger:
    logging.basicConfig(level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    )
    return structlog.get_logger()


log = configure_logging()
