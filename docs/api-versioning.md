# API Versioning Strategy

## Overview

LegalAssist AI uses URI-based versioning with Accept-Header fallback for backward-compatible API evolution. The versioning infrastructure is implemented in `api/versioning/` with OpenAPI specs at `api/specs/`.

## Versioning Strategy

### URI-Based Versioning (Primary)

```
GET /api/v1/cases/
GET /api/v2/cases/
POST /api/v1/documents/upload
```

The version is embedded in the URL path before the resource name.

### Accept-Header Versioning (Fallback)

For clients that cannot change URLs easily, version can be specified via the `Accept` header:

```
Accept: application/vnd.legalassist.v2+json
```

## Version Lifecycle

| Stage | Description | Timeline |
|-------|-------------|----------|
| Alpha | Internal testing, may change without notice | No backward compatibility |
| Beta | Public testing, breaking changes with 30-day notice | 1-2 months |
| Stable (v1) | Production ready, no breaking changes | Current supported version |
| Deprecated | Still functional but will be removed | At least 6 months from deprecation notice |
| Sunset | Removed from the API | After deprecation period |

## When to Create a New Version

- **Major**: Breaking changes to request/response schemas
- **Minor**: Adding new fields (backward compatible, no new version needed)
- **Patch**: Bug fixes (no version change)

## Adding a New API Version

1. Create the OpenAPI spec at `api/specs/v{N}.yaml`
2. Register the router in `api/versioning/router.py`
3. Maintain backward compatibility for existing clients
4. Add version migration guide to changelog
5. Update API documentation

## Version Router

The `api/versioning/router.py` dispatches requests based on version:

```python
# Request flow:
# GET /api/v2/cases/123
# 1. Router extracts version "2" from URL
# 2. Routes to v2 handler
# 3. If v2 handler not found, falls back to v1
# 4. Returns response with version header: X-API-Version: 2
```

## Backward Compatibility Rules

1. Never remove or rename existing fields
2. New fields must be optional with sensible defaults
3. Endpoint URLs must not change meaning
4. Error codes and formats must remain consistent
5. Pagination parameters must remain compatible

## Migration Guide

### v1 to v2 Migration

**Breaking Changes:**
- Response envelope changed from `{data: ...}` to `{data, meta, links}`
- Pagination now uses cursor-based instead of offset-based
- Date formats now use ISO 8601 with timezone

**Migration Steps:**
1. Update `Accept` header to `application/vnd.legalassist.v2+json`
2. Update pagination logic to use `cursor` and `limit` parameters
3. Parse new response envelope structure

## Testing

Test version-specific behavior with:

```python
from api.versioning.router import get_version_from_request

def test_version_routing():
    request = Request({"type": "http", "path": "/api/v2/cases/"})
    version = get_version_from_request(request)
    assert version == 2
```

See `tests/test_api_versioning.py` for comprehensive test cases.
