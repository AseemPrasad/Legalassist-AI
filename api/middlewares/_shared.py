from __future__ import annotations

SKIP_PATHS = {"/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live", "/metrics", "/"}
UPLOAD_PATH_PREFIXES = (
    "/api/v1/analyze/upload",
    "/api/v1/analyze/document",
    "/api/v1/documents",
)
ANALYTICS_PATH_PREFIXES = (
    "/api/v1/analytics",
)
IDEMPOTENT_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
