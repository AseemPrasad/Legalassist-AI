"""
Integration tests for PostgreSQL Row-Level Security (RLS) applied to API sessions.

Tests cover:
1. get_db_rls dependency applies RLS context for authenticated users (PostgreSQL).
2. get_db_rls dependency is a no-op on SQLite (local dev safety).
3. get_db_rls_optional applies RLS when a user is present, skips when absent.
4. RLS context is cleared after the request (teardown), even on exception.
5. API route source files use get_db_rls instead of bare get_db.
6. Non-numeric user IDs are handled gracefully (no crash).
"""

from __future__ import annotations

import os
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so tests run without the full stack
# ---------------------------------------------------------------------------

def _stub_module(name: str):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

for _mod in [
    "prometheus_client",
    "opentelemetry",
    "opentelemetry.trace",
    "opentelemetry.context",
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.instrumentation",
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.instrumentation.sqlalchemy",
    "opentelemetry.instrumentation.redis",
    "structlog",
]:
    _stub_module(_mod)

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain(gen: Generator):
    """Advance a generator dependency to yield and return the yielded value."""
    return next(gen)


def _make_request_stub():
    req = MagicMock()
    req.state = MagicMock()
    return req


def _make_user(user_id: str = "42"):
    """Return a minimal CurrentUser-like object."""
    user = MagicMock()
    user.user_id = user_id
    return user


# ---------------------------------------------------------------------------
# Unit tests for get_db_rls
# ---------------------------------------------------------------------------


class TestGetDbRlsUnit:
    """Unit tests that mock db.session internals to verify RLS calls."""

    def test_applies_rls_context_on_postgres(self):
        """apply_rls_context must be called with the authenticated user's ID."""
        from api.dependencies import get_db_rls

        request = _make_request_stub()
        user = _make_user("42")

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", True),
            patch("api.dependencies.apply_rls_context") as mock_apply,
            patch("api.dependencies.clear_rls_context") as mock_clear,
        ):
            gen = get_db_rls(request, user)
            db = _drain(gen)

            mock_apply.assert_called_once_with(mock_session, 42)
            assert db is mock_session

            # Exhaust the generator (simulates end of request)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_clear.assert_called_once_with(mock_session)
            mock_session.close.assert_called_once()

    def test_no_rls_on_sqlite(self):
        """apply_rls_context must NOT be called when _is_postgres is False."""
        from api.dependencies import get_db_rls

        request = _make_request_stub()
        user = _make_user("42")

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", False),
            patch("api.dependencies.apply_rls_context") as mock_apply,
            patch("api.dependencies.clear_rls_context") as mock_clear,
        ):
            gen = get_db_rls(request, user)
            _drain(gen)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_apply.assert_not_called()
            mock_clear.assert_not_called()
            mock_session.close.assert_called_once()

    def test_clears_rls_context_on_exception(self):
        """clear_rls_context must be called in the finally block even if the
        route handler raises an exception."""
        from api.dependencies import get_db_rls

        request = _make_request_stub()
        user = _make_user("7")

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", True),
            patch("api.dependencies.apply_rls_context"),
            patch("api.dependencies.clear_rls_context") as mock_clear,
        ):
            gen = get_db_rls(request, user)
            _drain(gen)

            # Simulate an exception thrown into the generator
            try:
                gen.throw(RuntimeError("route error"))
            except RuntimeError:
                pass

            mock_clear.assert_called_once()
            mock_session.close.assert_called_once()

    def test_non_numeric_user_id_skips_rls(self):
        """A non-numeric user_id must not crash; RLS is simply skipped."""
        from api.dependencies import get_db_rls

        request = _make_request_stub()
        user = _make_user("not-a-number")

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", True),
            patch("api.dependencies.apply_rls_context") as mock_apply,
            patch("api.dependencies.clear_rls_context"),
        ):
            gen = get_db_rls(request, user)
            _drain(gen)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_apply.assert_not_called()

    def test_session_always_closed(self):
        """Session.close() must be called regardless of whether RLS is used."""
        from api.dependencies import get_db_rls

        request = _make_request_stub()
        user = _make_user("5")

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", True),
            patch("api.dependencies.apply_rls_context"),
            patch("api.dependencies.clear_rls_context"),
        ):
            gen = get_db_rls(request, user)
            _drain(gen)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_session.close.assert_called_once()


class TestGetDbRlsOptionalUnit:
    """Unit tests for get_db_rls_optional."""

    def test_applies_rls_when_user_present(self):
        from api.dependencies import get_db_rls_optional

        request = _make_request_stub()
        user = _make_user("99")

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", True),
            patch("api.dependencies.apply_rls_context") as mock_apply,
            patch("api.dependencies.clear_rls_context"),
        ):
            gen = get_db_rls_optional(request, user)
            _drain(gen)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_apply.assert_called_once_with(mock_session, 99)

    def test_skips_rls_when_no_user(self):
        """Unauthenticated requests must not set any RLS context."""
        from api.dependencies import get_db_rls_optional

        request = _make_request_stub()

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", True),
            patch("api.dependencies.apply_rls_context") as mock_apply,
            patch("api.dependencies.clear_rls_context"),
        ):
            gen = get_db_rls_optional(request, None)
            _drain(gen)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_apply.assert_not_called()

    def test_session_closed_when_no_user(self):
        from api.dependencies import get_db_rls_optional

        request = _make_request_stub()

        mock_session = MagicMock()
        mock_session_local = MagicMock(return_value=mock_session)

        with (
            patch("api.dependencies.SessionLocal", mock_session_local),
            patch("api.dependencies._is_postgres", False),
            patch("api.dependencies.apply_rls_context"),
            patch("api.dependencies.clear_rls_context"),
        ):
            gen = get_db_rls_optional(request, None)
            _drain(gen)
            try:
                next(gen)
            except StopIteration:
                pass

            mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Source-level checks: route files must use get_db_rls
# ---------------------------------------------------------------------------


class TestRoutesUseGetDbRls:
    """Verify that route modules import and use get_db_rls, not bare get_db."""

    def _source(self, module_path: str) -> str:
        with open(module_path, encoding="utf-8") as f:
            return f.read()

    def _route(self, name: str) -> str:
        base = os.path.join(os.path.dirname(__file__), "..", "api", "routes")
        return os.path.join(base, name)

    def test_cases_route_uses_get_db_rls(self):
        src = self._source(self._route("cases.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_deadlines_route_uses_get_db_rls(self):
        src = self._source(self._route("deadlines.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_auth_route_uses_get_db_rls(self):
        src = self._source(self._route("auth.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_reports_route_uses_get_db_rls(self):
        src = self._source(self._route("reports.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_analytics_route_uses_get_db_rls(self):
        src = self._source(self._route("analytics.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_audit_route_uses_get_db_rls(self):
        src = self._source(self._route("audit.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_knowledge_route_uses_get_db_rls(self):
        src = self._source(self._route("knowledge.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_case_search_route_uses_get_db_rls(self):
        src = self._source(self._route("case_search.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_models_route_uses_get_db_rls(self):
        src = self._source(self._route("models.py"))
        assert "get_db_rls" in src
        assert "Depends(get_db)" not in src

    def test_dependencies_exports_get_db_rls(self):
        dep_path = os.path.join(
            os.path.dirname(__file__), "..", "api", "dependencies.py"
        )
        src = self._source(dep_path)
        assert "def get_db_rls(" in src
        assert "def get_db_rls_optional(" in src
        assert "apply_rls_context" in src
        assert "clear_rls_context" in src

    def test_websocket_applies_rls(self):
        ws_path = os.path.join(
            os.path.dirname(__file__), "..", "api", "websockets", "case_timeline.py"
        )
        src = self._source(ws_path)
        assert "apply_rls_context" in src
        assert "clear_rls_context" in src

    def test_setup_rls_documents_migration_order(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "setup_rls.py"
        )
        src = self._source(script_path)
        assert "after" in src.lower() and "migrat" in src.lower(), (
            "setup_rls.py must document that it runs after migrations"
        )


# ---------------------------------------------------------------------------
# PostgreSQL integration tests (skipped when no PG available)
# ---------------------------------------------------------------------------

def _pg_available() -> bool:
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url.startswith("postgresql"):
        return False
    try:
        from sqlalchemy import create_engine, text

        eng = create_engine(db_url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_available(), reason="PostgreSQL not available")
class TestRlsPostgresIntegration:
    """End-to-end tests that require a live PostgreSQL connection.

    Verifies that apply_rls_context / clear_rls_context actually set and
    reset the ``app.current_user_id`` session variable in PostgreSQL, and
    that the variable is not shared across separate connections.
    """

    @pytest.fixture()
    def pg_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(os.environ["DATABASE_URL"])
        Session = sessionmaker(bind=engine)
        db = Session()
        yield db
        db.close()

    def test_apply_rls_context_sets_variable(self, pg_session):
        from sqlalchemy import text
        from db.session import apply_rls_context

        apply_rls_context(pg_session, 123)
        result = pg_session.execute(
            text("SELECT current_setting('app.current_user_id', true)")
        ).scalar()
        assert result == "123", f"Expected '123', got {result!r}"

    def test_clear_rls_context_resets_variable(self, pg_session):
        from sqlalchemy import text
        from db.session import apply_rls_context, clear_rls_context

        apply_rls_context(pg_session, 456)
        clear_rls_context(pg_session)
        result = pg_session.execute(
            text("SELECT current_setting('app.current_user_id', true)")
        ).scalar()
        assert result in (None, ""), f"Expected empty after reset, got {result!r}"

    def test_rls_context_not_shared_across_connections(self):
        """Two separate sessions must not share the app.current_user_id variable."""
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        from db.session import apply_rls_context

        engine = create_engine(os.environ["DATABASE_URL"])
        Session = sessionmaker(bind=engine)

        db1 = Session()
        db2 = Session()
        try:
            apply_rls_context(db1, 111)
            result = db2.execute(
                text("SELECT current_setting('app.current_user_id', true)")
            ).scalar()
            assert result in (None, ""), (
                f"Session isolation violated: db2 saw user_id={result!r}"
            )
        finally:
            db1.close()
            db2.close()

    def test_get_db_rls_sets_and_clears_variable(self):
        """Full lifecycle: get_db_rls sets the variable during yield and clears it after."""
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        from api.dependencies import get_db_rls

        engine = create_engine(os.environ["DATABASE_URL"])
        _SessionLocal = sessionmaker(bind=engine)

        request = _make_request_stub()
        user = _make_user("77")

        with (
            patch("api.dependencies.SessionLocal", _SessionLocal),
            patch("api.dependencies._is_postgres", True),
        ):
            gen = get_db_rls(request, user)
            db = _drain(gen)

            result = db.execute(
                text("SELECT current_setting('app.current_user_id', true)")
            ).scalar()
            assert result == "77", f"Expected '77', got {result!r}"

            try:
                next(gen)
            except StopIteration:
                pass

            # Open a fresh session to confirm the variable was reset
            db2 = _SessionLocal()
            try:
                result2 = db2.execute(
                    text("SELECT current_setting('app.current_user_id', true)")
                ).scalar()
                assert result2 in (None, ""), (
                    f"RLS context not cleared after teardown: {result2!r}"
                )
            finally:
                db2.close()
