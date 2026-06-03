"""
Client creation and concurrency management for CLI.

This module handles:
- OpenAI/OpenRouter client initialization
- API semaphore management for concurrency control
- Chat completion with retry logic and exponential backoff
"""

import os
import time
import threading
from typing import Optional, Tuple

from openai import OpenAI, RateLimitError
import structlog

from config import Config

LOGGER = structlog.get_logger(__name__)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4-turbo")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Global semaphore for API concurrency control
_API_SEMAPHORE: Optional[threading.Semaphore] = None
_SEMAPHORE_LOCK = threading.Lock()


class CLIError(Exception):
    """Base exception for CLI operations."""
    pass


def reinitialize_semaphore(concurrency: int) -> None:
    """
    Replace the global API semaphore with a new one sized to *concurrency*.

    Calling this before any worker threads are spawned ensures the correct
    limit is applied regardless of whether execution entered through main()
    or directly via process_command / batch_command (e.g. in tests).
    
    Args:
        concurrency: Maximum number of concurrent API calls allowed
    """
    global _API_SEMAPHORE
    with _SEMAPHORE_LOCK:
        _API_SEMAPHORE = threading.Semaphore(concurrency)


def get_api_semaphore() -> threading.Semaphore:
    """
    Get the API semaphore, initializing it lazily with default concurrency if needed.
    
    Returns:
        The global API semaphore for concurrency control
    """
    global _API_SEMAPHORE
    if _API_SEMAPHORE is None:
        with _SEMAPHORE_LOCK:
            if _API_SEMAPHORE is None:
                _API_SEMAPHORE = threading.Semaphore(5)
    return _API_SEMAPHORE


def get_client() -> OpenAI:
    """
    Create and return an OpenAI client configured for the current environment.
    
    Uses OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY from environment.
    Base URL defaults to OpenRouter but can be overridden with OPENROUTER_BASE_URL.
    
    Returns:
        Configured OpenAI client instance
        
    Raises:
        CLIError: If no API key is found in environment
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise CLIError(
            "Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY in your environment. "
            "You can also add these to your .env file."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def get_usage_tokens(response) -> Tuple[int, int, int]:
    """
    Extract token counts from an OpenAI API response.
    
    Args:
        response: OpenAI API response object
        
    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_tokens)
    """
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1k: float,
    completion_cost_per_1k: float,
) -> float:
    """
    Estimate the USD cost of an API call based on token counts and rates.
    
    Args:
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        prompt_cost_per_1k: Cost per 1000 prompt tokens (USD)
        completion_cost_per_1k: Cost per 1000 completion tokens (USD)
        
    Returns:
        Estimated cost in USD
    """
    return ((prompt_tokens / 1000.0) * prompt_cost_per_1k) + (
        (completion_tokens / 1000.0) * completion_cost_per_1k
    )


def chat_completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 5,
    timeout: Optional[float] = None,
):
    """
    Perform a chat completion request with retries and concurrency control.
    
    Features:
    - Concurrency control via global semaphore
    - Exponential backoff for rate limiting (429 errors)
    - Timeout management
    - Detailed debug logging
    
    Args:
        client: OpenAI client instance
        model: Model name to use
        system_prompt: System message for the LLM
        user_prompt: User message for the LLM
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Maximum retry attempts for rate limits
        timeout: Request timeout in seconds (defaults to Config.LLM_TIMEOUT)
        
    Returns:
        OpenAI API response object
        
    Raises:
        RateLimitError: If rate limit is hit after max_retries
        Other OpenAI exceptions for other API errors
    """
    if timeout is None:
        timeout = Config.LLM_TIMEOUT
    
    last_err = None
    
    LOGGER.debug(
        "chat_completion_start",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout
    )
    
    for attempt in range(max_retries):
        try:
            # Concurrency control to prevent overwhelming the API
            with get_api_semaphore():
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                
                LOGGER.debug(
                    "chat_completion_success",
                    model=model,
                    attempt=attempt + 1
                )
                return response
                
        except RateLimitError as e:
            last_err = e
            if attempt == max_retries - 1:
                LOGGER.error("api_rate_limit_exhausted", attempts=max_retries, error=str(e))
                raise
            
            # Exponential backoff: 2, 4, 8, 16, 32 seconds
            wait_time = 2 ** (attempt + 1)
            LOGGER.warning(
                "api_rate_limited",
                attempt=attempt + 1,
                wait_seconds=wait_time,
                error=str(e)
            )
            time.sleep(wait_time)
        except Exception as e:
            # Don't retry on other errors (auth, invalid params)
            LOGGER.debug("chat_completion_fatal_error", error=str(e), error_type=type(e).__name__)
            raise
    
    if last_err:
        raise last_err
