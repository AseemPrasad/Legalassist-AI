"""Verify all core project modules import cleanly without errors."""

import importlib
import pkgutil
import pytest
import sys


def _iter_project_packages():
    """Yield all top-level packages under the project directory."""
    project_packages = [
        "api",
        "core",
        "db",
        "services",
        "domain",
        "observability",
        "notifications",
    ]
    for pkg_name in project_packages:
        try:
            importlib.import_module(pkg_name)
            yield pkg_name
        except ImportError:
            pass


def _iter_package_modules(package_name):
    """Yield all module names under a given package."""
    try:
        package = importlib.import_module(package_name)
        path = getattr(package, "__path__", [])
        for importer, modname, ispkg in pkgutil.walk_packages(
            path, prefix=f"{package_name}.",
        ):
            yield modname
    except ImportError:
        pass


MODULES_TO_SKIP = {
    "core.all_translations",
    "core.e2ee",
    "core.secrets",
    "core.kms",
    "api.specs",
}


@pytest.mark.parametrize("module_name", [
    m for pkg in _iter_project_packages()
    for m in _iter_package_modules(pkg)
    if m not in MODULES_TO_SKIP
])
def test_module_imports(module_name):
    """Assert that each module imports without raising ImportError or NameError."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"Failed to import {module_name}: {exc}")
    except NameError as exc:
        pytest.fail(f"NameError during import of {module_name}: {exc}")
    except Exception as exc:
        pytest.fail(f"Unexpected exception during import of {module_name}: {type(exc).__name__}: {exc}")


def test_critical_modules_import():
    """Verify critical top-level modules that are commonly imported."""
    critical_modules = [
        "api.auth",
        "api.config",
        "api.csrf",
        "api.dependencies",
        "api.errors",
        "api.jwt_auth",
        "api.limiter",
        "api.middleware",
        "api.validation",
        "core.case_search_engine",
        "core.citation_engine",
        "core.deadline_engine",
        "core.rag_engine",
        "core.vector_store",
        "core.app_utils",
        "core.argument_extraction_engine",
        "core.precedent_matcher",
        "core.document_verifier",
        "db.session",
        "db.base",
        "db.crud.cases",
        "db.crud.users",
        "db.crud.notifications",
        "db.crud.audit",
        "api.versioning.router",
        "api.websockets.connection_manager",
        "api.websockets.reconnection",
        "api.websockets.router",
        "core.event_store",
        "core.domain_events",
    ]
    for mod_name in critical_modules:
        try:
            importlib.import_module(mod_name)
        except ImportError as exc:
            pytest.fail(f"Critical module {mod_name} failed to import: {exc}")
        except Exception as exc:
            pytest.fail(f"Critical module {mod_name} raised {type(exc).__name__}: {exc}")


def test_config_imports():
    """Verify that config and settings can be loaded."""
    from api.config import get_settings
    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, "API_TITLE")
    assert hasattr(settings, "API_VERSION")


def test_database_session_imports():
    """Verify database session imports work correctly."""
    from db.session import SessionLocal, engine, Base
    assert SessionLocal is not None
    assert engine is not None
    assert Base is not None
