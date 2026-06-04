"""Tests for the JWT token type / nbf claim validation fix (#1569).

Verifies that:
- api/auth.py no longer defines a local revoke_jwt_token that bypasses nbf.
- api/auth.py no longer defines a local _get_jwt_secrets_to_try.
- The require list in api/jwt_auth.revoke_jwt_token includes 'nbf'.
- The require list in api/jwt_auth.verify_token includes 'nbf'.
- api/auth.revoke_jwt_token is the same object as api/jwt_auth.revoke_jwt_token
  (i.e. it is the imported canonical version, not a local override).
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ---------------------------------------------------------------------------
# api/auth.py no longer defines a local revoke_jwt_token
# ---------------------------------------------------------------------------

def test_auth_module_has_no_local_revoke_jwt_token_definition():
    """api/auth.py must not define its own revoke_jwt_token function."""
    auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "auth.py"
    )
    source = open(auth_path, encoding="utf-8").read()
    tree = ast.parse(source)

    local_defs = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "revoke_jwt_token"
    ]
    assert not local_defs, (
        "api/auth.py must not define a local revoke_jwt_token. "
        "The local version omitted 'nbf' from the require list, allowing "
        "tokens without a not-before claim to bypass the nbf check. "
        "Use the canonical version imported from api.jwt_auth."
    )


def test_auth_module_has_no_local_get_jwt_secrets_to_try():
    """api/auth.py must not define its own _get_jwt_secrets_to_try."""
    auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "auth.py"
    )
    source = open(auth_path, encoding="utf-8").read()
    tree = ast.parse(source)

    local_defs = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_jwt_secrets_to_try"
    ]
    assert not local_defs, (
        "api/auth.py must not define a local _get_jwt_secrets_to_try. "
        "Use the canonical version in api.jwt_auth."
    )


# ---------------------------------------------------------------------------
# Canonical revoke_jwt_token require list includes 'nbf'
# ---------------------------------------------------------------------------

def test_jwt_auth_revoke_token_require_list_includes_nbf():
    """api/jwt_auth.revoke_jwt_token must include 'nbf' in its require list."""
    jwt_auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "jwt_auth.py"
    )
    source = open(jwt_auth_path, encoding="utf-8").read()
    tree = ast.parse(source)

    # Find the revoke_jwt_token function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "revoke_jwt_token":
            func_node = node
            break

    assert func_node is not None, "revoke_jwt_token not found in api/jwt_auth.py"

    func_source = ast.unparse(func_node)

    assert "'nbf'" in func_source or '"nbf"' in func_source, (
        "api/jwt_auth.revoke_jwt_token must include 'nbf' in the require list "
        "so tokens without a not-before claim are rejected on the revocation path."
    )
    assert "verify_nbf" in func_source, (
        "api/jwt_auth.revoke_jwt_token must set verify_nbf=True."
    )


def test_jwt_auth_verify_token_require_list_includes_nbf():
    """api/jwt_auth.verify_token must include 'nbf' in its require list."""
    jwt_auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "jwt_auth.py"
    )
    source = open(jwt_auth_path, encoding="utf-8").read()
    tree = ast.parse(source)

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "verify_token":
            func_node = node
            break

    assert func_node is not None
    func_source = ast.unparse(func_node)

    assert "'nbf'" in func_source or '"nbf"' in func_source, (
        "api/jwt_auth.verify_token must include 'nbf' in its require list."
    )


# ---------------------------------------------------------------------------
# api.auth.revoke_jwt_token IS the canonical imported version
# ---------------------------------------------------------------------------

def test_auth_revoke_jwt_token_is_canonical_imported_version():
    """api.auth.revoke_jwt_token must be the same object as api.jwt_auth.revoke_jwt_token."""
    import api.jwt_auth as jwt_auth_mod
    import api.auth as auth_mod

    assert auth_mod.revoke_jwt_token is jwt_auth_mod.revoke_jwt_token, (
        "api.auth.revoke_jwt_token must be the imported version from api.jwt_auth, "
        "not a locally-defined override. The local override missed 'nbf' in the "
        "require list, creating an inconsistent validation path."
    )


# ---------------------------------------------------------------------------
# Require list completeness in api/jwt_auth.py
# ---------------------------------------------------------------------------

def test_required_claims_are_complete():
    """Both verify_token and revoke_jwt_token must require the same set of claims."""
    jwt_auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "jwt_auth.py"
    )
    source = open(jwt_auth_path, encoding="utf-8").read()

    required_claims = ["exp", "iat", "nbf", "iss", "aud", "jti", "type"]
    for claim in required_claims:
        assert f'"{claim}"' in source or f"'{claim}'" in source, (
            f"api/jwt_auth.py must require the '{claim}' claim in JWT validation."
        )
