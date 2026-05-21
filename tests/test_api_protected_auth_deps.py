import inspect
from api.main import create_app

PROTECTED_DEP_NAMES = {"get_current_user", "get_admin_user", "get_attorney_user"}


def _collect_dependency_calls(route):
    calls = []
    dependant = getattr(route, "dependant", None)
    if not dependant:
        return calls
    for dep in getattr(dependant, "dependencies", []) or []:
        call = getattr(dep, "call", None)
        if call and inspect.isfunction(call):
            calls.append(call)
    return calls


def test_api_protected_routes_use_api_auth():
    """Ensure any route depending on auth uses the canonical `api.auth` module.

    This guards against drift where some routes import auth helpers
    from the top-level `auth` module instead of the API-specific
    `api.auth` implementation.
    """
    app = create_app()
    mismatches = []

    for route in app.routes:
        calls = _collect_dependency_calls(route)
        for call in calls:
            if call.__name__ in PROTECTED_DEP_NAMES:
                if not (call.__module__ or "").startswith("api.auth"):
                    mismatches.append(f"{route.path} -> {call.__module__}.{call.__name__}")

    assert not mismatches, "Protected routes using non-api.auth dependencies:\n" + "\n".join(mismatches)
