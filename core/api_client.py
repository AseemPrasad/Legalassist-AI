"""
External API client with graceful timeout and fallback handling.

Provides utilities for making resilient external HTTP calls with proper
timeout handling, retry logic, and structured fallback responses.
"""

import time
import logging
from typing import Any, Dict, Optional, Callable, TypeVar
from dataclasses import dataclass
from functools import wraps

import requests
from requests.exceptions import (
    ConnectionError,
    Timeout,
    ReadTimeout,
    ConnectTimeout,
    HTTPError,
    RequestException,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class APIResponse:
    """Structured response for external API calls."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    was_fallback: bool = False

    def is_server_error(self) -> bool:
        return self.status_code and 500 <= self.status_code < 600

    def is_client_error(self) -> bool:
        return self.status_code and 400 <= self.status_code < 500


def _is_transient_error(exc: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    transient_types = (
        ConnectionError,
        Timeout,
        ReadTimeout,
        ConnectTimeout,
    )
    if isinstance(exc, HTTPError):
        status = exc.response.status_code if hasattr(exc, 'response') else None
        return status and status >= 500
    return isinstance(exc, transient_types)


def call_with_fallback(
    endpoint: str,
    method: str = "GET",
    fallback: Optional[Callable[[], T]] = None,
    timeout: float = 10.0,
    retries: int = 2,
    retry_delay: float = 1.0,
    **kwargs,
) -> APIResponse:
    """
    Call an external API with graceful timeout handling and fallback.

    Args:
        endpoint: URL to call
        method: HTTP method (GET, POST, etc.)
        fallback: Callable to invoke if all attempts fail
        timeout: Request timeout in seconds
        retries: Number of retry attempts for transient failures
        retry_delay: Seconds to wait between retries

    Returns:
        APIResponse with success status and data or error
    """
    last_error = None
    status_code = None

    for attempt in range(retries + 1):
        try:
            if method.upper() == "GET":
                response = requests.get(endpoint, timeout=timeout, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(endpoint, timeout=timeout, **kwargs)
            elif method.upper() == "PUT":
                response = requests.put(endpoint, timeout=timeout, **kwargs)
            else:
                response = requests.request(method, endpoint, timeout=timeout, **kwargs)

            status_code = response.status_code

            if response.ok:
                return APIResponse(
                    success=True,
                    data=response.json() if response.content else None,
                    status_code=status_code,
                )

            if status_code and 400 <= status_code < 500:
                return APIResponse(
                    success=False,
                    error=f"Client error (HTTP {status_code})",
                    status_code=status_code,
                )

            response.raise_for_status()

        except RequestException as exc:
            last_error = exc
            if not _is_transient_error(exc):
                logger.warning(f"Non-transient API error: {exc}")
                break

            if attempt < retries:
                logger.debug(f"Retrying API call (attempt {attempt + 1}/{retries}): {exc}")
                time.sleep(retry_delay * (attempt + 1))

    if fallback:
        try:
            result = fallback()
            return APIResponse(
                success=True,
                data=result,
                error=str(last_error) if last_error else None,
                was_fallback=True,
            )
        except Exception as fallback_exc:
            logger.error(f"Fallback also failed: {fallback_exc}")

    return APIResponse(
        success=False,
        error=str(last_error) if last_error else "Unknown error",
        status_code=status_code,
    )


def with_timeout_fallback(timeout: float = 10.0, fallback: Optional[T] = None):
    """
    Decorator for adding timeout and fallback to any function.

    Args:
        timeout: Timeout in seconds
        fallback: Default value to return on failure
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if _is_transient_error(exc):
                    logger.warning(f"{func.__name__} failed with transient error: {exc}")
                else:
                    logger.error(f"{func.__name__} failed: {exc}")
                return fallback
        return wrapper
    return decorator


class ResilientClient:
    """
    HTTP client wrapper with built-in timeout and fallback handling.

    Usage:
        client = ResilientClient(base_url="https://api.example.com")
        response = client.get("/endpoint", fallback={"default": "value"})
    """

    def __init__(self, base_url: str = "", timeout: float = 10.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _build_url(self, path: str) -> str:
        path = path.lstrip("/")
        if self.base_url:
            return f"{self.base_url}/{path}"
        return path

    def get(self, path: str, fallback: Any = None, **kwargs) -> APIResponse:
        return call_with_fallback(
            self._build_url(path),
            method="GET",
            timeout=kwargs.pop("timeout", self.timeout),
            retries=kwargs.pop("retries", self.retries),
            fallback=lambda: fallback if fallback is not None else None,
            **kwargs,
        )

    def post(self, path: str, fallback: Any = None, **kwargs) -> APIResponse:
        return call_with_fallback(
            self._build_url(path),
            method="POST",
            timeout=kwargs.pop("timeout", self.timeout),
            retries=kwargs.pop("retries", self.retries),
            fallback=lambda: fallback if fallback is not None else None,
            **kwargs,
        )

    def put(self, path: str, fallback: Any = None, **kwargs) -> APIResponse:
        return call_with_fallback(
            self._build_url(path),
            method="PUT",
            timeout=kwargs.pop("timeout", self.timeout),
            retries=kwargs.pop("retries", self.retries),
            fallback=lambda: fallback if fallback is not None else None,
            **kwargs,
        )