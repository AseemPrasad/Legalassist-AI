from __future__ import annotations

from fastapi import FastAPI

from .idempotency import http_idempotency_manager, idempotency_middleware, is_safe_to_cache
from .rate_limit import rate_limit_middleware, settings as rate_limit_settings
from .request_size import request_size_limit_middleware


def register_middlewares(app: FastAPI) -> None:
    from api.middleware import add_correlation_id_middleware, error_handling_middleware, logging_middleware

    app.middleware("http")(request_size_limit_middleware)
    app.middleware("http")(idempotency_middleware)
    app.middleware("http")(logging_middleware)
    app.middleware("http")(add_correlation_id_middleware)
    app.middleware("http")(error_handling_middleware)

    if rate_limit_settings.RATE_LIMIT_ENABLED:
        app.middleware("http")(rate_limit_middleware)


__all__ = [
    "http_idempotency_manager",
    "idempotency_middleware",
    "is_safe_to_cache",
    "rate_limit_middleware",
    "register_middlewares",
    "request_size_limit_middleware",
]
