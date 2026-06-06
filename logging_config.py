import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

import structlog

try:
    from rich.logging import RichHandler
except ModuleNotFoundError:
    RichHandler = None

# Default constants for logging configuration
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 10
DEFAULT_LOG_FORMAT = "%(message)s"


def ensure_log_directory(log_dir: str) -> None:
    """
    Ensure the directory for log files exists.
    
    Args:
        log_dir: The path to the logging directory.
    """
    path = Path(log_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_console_handler(level: int = logging.INFO) -> logging.Handler:
    """
    Create a console handler, preferably a RichHandler if available.
    
    Args:
        level: Logging level for this handler.
        
    Returns:
        A configured logging handler.
    """
    if RichHandler:
        handler = RichHandler(
            rich_tracebacks=True, 
            markup=True, 
            show_time=True,
            show_path=False
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        
    handler.setLevel(level)
    return handler


def get_rotating_file_handler(
    filename: str, 
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT
) -> logging.handlers.RotatingFileHandler:
    """
    Create a RotatingFileHandler to prevent log files from growing indefinitely.
    
    This handler limits the size of each log file and keeps a configured 
    number of backups, which is crucial for preventing disk exhaustion.
    
    Args:
        filename: Name/path of the log file.
        level: Logging level.
        max_bytes: Maximum size in bytes before rotating.
        backup_count: Number of backup files to keep.
        
    Returns:
        A configured RotatingFileHandler.
    """
    handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    return handler


def get_timed_rotating_file_handler(
    filename: str,
    level: int = logging.INFO,
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = DEFAULT_BACKUP_COUNT
) -> logging.handlers.TimedRotatingFileHandler:
    """
    Create a TimedRotatingFileHandler for time-based log rotation.
    
    Args:
        filename: Name/path of the log file.
        level: Logging level.
        when: Type of interval ('s', 'm', 'h', 'd', 'midnight', 'w0'-'w6').
        interval: Interval count.
        backup_count: Number of backup files to keep.
        
    Returns:
        A configured TimedRotatingFileHandler.
    """
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=filename,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    return handler


def configure_structlog(level: int) -> None:
    """
    Configure structlog processors and behaviors.
    
    Args:
        level: Logging level.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(indent=None, sort_keys=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def configure_logging(
    level: int = logging.INFO,
    log_dir: str = DEFAULT_LOG_DIR,
    log_file_prefix: str = "app",
    enable_rotation: bool = True,
    use_timed_rotation: bool = False
) -> None:
    """
    Configure logging for the application.
    
    Sets up both standard logging and structlog. It can emit structured JSON 
    via structlog and uses RichHandler for human-friendly console output 
    during development. Also includes log rotation to prevent indefinite 
    growth of log files.
    
    Args:
        level: The minimum logging level to capture.
        log_dir: Directory where log files should be stored.
        log_file_prefix: Prefix for log file names.
        enable_rotation: Whether to add a file handler with rotation.
        use_timed_rotation: If True, uses TimedRotatingFileHandler; 
                            otherwise uses RotatingFileHandler (size based).
    """
    handlers: List[logging.Handler] = [get_console_handler(level)]
    
    if enable_rotation:
        ensure_log_directory(log_dir)
        log_filename = os.path.join(log_dir, f"{log_file_prefix}.log")
        
        if use_timed_rotation:
            file_handler = get_timed_rotating_file_handler(
                filename=log_filename,
                level=level
            )
        else:
            file_handler = get_rotating_file_handler(
                filename=log_filename,
                level=level
            )
            
        handlers.append(file_handler)

    # Configure the standard logging module
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True
    )
    
    # Configure structlog
    configure_structlog(level)


def get_sensitive_data_redactor():
    """
    Returns a structlog processor that automatically redacts sensitive keys 
    such as passwords, tokens, API keys, and private customer details 
    from log records to prevent security leaks in compliance with GDPR.
    """
    sensitive_keys = {"password", "token", "secret", "authorization", "ssn", "credit_card", "key"}
    def processor(logger, name, event_dict):
        for key in list(event_dict.keys()):
            if any(s in key.lower() for s in sensitive_keys):
                event_dict[key] = "[REDACTED]"
        return event_dict
    return processor
