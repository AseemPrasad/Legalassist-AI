"""
Role-Based Access Control (RBAC) for LegalAssist AI

Role Hierarchy:
    admin > attorney > paralegal > client

Permission Matrix:
    case:read        - View case details and timeline
    case:write       - Create/update cases
    case:delete      - Delete cases
    case:assign      - Assign cases to attorneys/paralegals
    document:read    - View documents
    document:write   - Upload/annotate documents
    document:delete  - Delete documents
    deadline:read    - View deadlines
    deadline:write   - Create/update deadlines
    deadline:delete  - Delete deadlines
    report:read      - View reports
    report:write     - Generate reports
    report:delete    - Delete reports
    user:read        - View user profiles
    user:write       - Modify user profiles
    user:manage      - Create/delete users (admin/attorney only)
    analytics:read   - View analytics
    admin:panel      - Access admin panel (admin only)
"""

from __future__ import annotations

import enum
from functools import wraps
from typing import Callable

from fastapi import Depends, HTTPException, status, Request

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "case:read", "case:write", "case:delete", "case:assign",
        "document:read", "document:write", "document:delete",
        "deadline:read", "deadline:write", "deadline:delete",
        "report:read", "report:write", "report:delete",
        "user:read", "user:write", "user:manage",
        "analytics:read", "admin:panel",
    },
    "attorney": {
        "case:read", "case:write",
        "document:read", "document:write",
        "deadline:read", "deadline:write",
        "report:read", "report:write",
        "user:read",
        "analytics:read",
    },
    "paralegal": {
        "case:read",
        "document:read", "document:write",
        "deadline:read", "deadline:write",
        "report:read",
        "user:read",
        "analytics:read",
    },
    "client": {
        "case:read",
        "document:read",
        "deadline:read",
        "report:read",
        "user:read",
    },
}


def get_permissions_for_role(role: str) -> set[str]:
    return ROLE_PERMISSIONS.get(role, set())


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())


def has_any_permission(role: str, permissions: list[str]) -> bool:
    role_perms = ROLE_PERMISSIONS.get(role, set())
    return any(p in role_perms for p in permissions)


def has_all_permissions(role: str, permissions: list[str]) -> bool:
    role_perms = ROLE_PERMISSIONS.get(role, set())
    return all(p in role_perms for p in permissions)


def get_user_role(user) -> str:
    if hasattr(user, "is_admin") and user.is_admin:
        return "admin"
    if hasattr(user, "role"):
        return user.role
    return "client"


def require_permission(*permissions: str, require_all: bool = False):
    """
    Dependency to enforce permission(s) on a route handler.

    Usage:
        @app.get("/cases")
        def list_cases(user: CurrentUser = Depends(require_permission("case:read"))):
            ...

        @app.delete("/cases/{id}")
        def delete_case(user: CurrentUser = Depends(require_permission("case:delete"))):
            ...

        @app.post("/users")
        def create_user(user: CurrentUser = Depends(require_permission("user:manage"))):
            ...
    """
    def dependency(current_user: "CurrentUser") -> "CurrentUser":
        role = current_user.role
        if require_all:
            if not has_all_permissions(role, list(permissions)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' lacks required permissions: {list(permissions)}",
                )
        else:
            if not has_any_permission(role, list(permissions)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' does not have any of the required permissions: {list(permissions)}",
                )
        return current_user

    return dependency


def require_role(*roles: str):
    """
    Dependency to enforce minimum role level.

    Usage:
        @app.get("/admin")
        def admin_panel(user: CurrentUser = Depends(require_role("admin"))):
            ...
    """
    def dependency(current_user: "CurrentUser") -> "CurrentUser":
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {list(roles)}",
            )
        return current_user

    return dependency


def require_min_role(min_role: str):
    """
    Dependency to enforce minimum role level hierarchically.

    Hierarchy: admin > attorney > paralegal > client
    """
    ROLE_HIERARCHY = ["client", "paralegal", "attorney", "admin"]

    def dependency(current_user: "CurrentUser") -> "CurrentUser":
        user_level = ROLE_HIERARCHY.index(current_user.role) if current_user.role in ROLE_HIERARCHY else 0
        min_level = ROLE_HIERARCHY.index(min_role) if min_role in ROLE_HIERARCHY else 0
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is below minimum required role '{min_role}'",
            )
        return current_user

    return dependency