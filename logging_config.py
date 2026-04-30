import logging

import structlog
try:
    from rich.logging import RichHandler
except ModuleNotFoundError:
    RichHandler = None


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application to emit structured JSON via structlog

    Uses RichHandler for human-friendly console output during development and
    a JSON renderer for structured logs.
    """

    handler = (
        RichHandler(rich_tracebacks=True, markup=True)
        if RichHandler
        else logging.StreamHandler()
    )

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
    )

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
