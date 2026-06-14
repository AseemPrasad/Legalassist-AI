"""
Unified Policy Engine for LegalAssist AI.

Centralizes all authorization decisions. Services and routes declare
requirements declaratively instead of imperative ownership checks.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PolicyDecision(enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


class PolicyType(enum.Enum):
    OWNERSHIP = "ownership"
    ROLE_ADMIN = "role_admin"
    ROLE_ATTORNEY = "role_attorney"
    ROLE_USER = "role_user"
    CASE_COLLABORATOR = "case_collaborator"
    RESOURCE_OWNER = "resource_owner"


@dataclass(frozen=True)
class Policy:
    policy_type: PolicyType
    resource_type: str
    action: str
    custom_evaluator: Optional[Callable[[UserContext, Any, Optional[Session]], PolicyDecision]] = None


@dataclass
class UserContext:
    user_id: int
    email: str
    role: str
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def is_admin(self) -> bool:
        return self.role == "admin" or self.extra.get("is_admin", False)

    def is_attorney(self) -> bool:
        return self.role in ("attorney", "admin") or self.is_admin()

    def is_api(self) -> bool:
        return self.role == "api"


class PolicyRegistry:
    def __init__(self):
        self._policies: Dict[tuple[str, str], List[Policy]] = {}

    def register(self, policy: Policy) -> None:
        key = (policy.resource_type, policy.action)
        self._policies.setdefault(key, []).append(policy)

    def get_policies(self, resource_type: str, action: str) -> List[Policy]:
        return self._policies.get((resource_type, action), [])


_registry = PolicyRegistry()


def register_policy(policy: Policy) -> None:
    _registry.register(policy)
    logger.debug(
        "policy_registered",
        resource_type=policy.resource_type,
        action=policy.action,
        policy_type=policy.policy_type.value,
    )


class PolicyEngine:
    def __init__(self, registry: Optional[PolicyRegistry] = None):
        self.registry = registry or _registry

    def evaluate(
        self,
        user: UserContext,
        resource_type: str,
        action: str,
        resource: Any = None,
        db: Optional[Session] = None,
    ) -> PolicyDecision:
        policies = self.registry.get_policies(resource_type, action)
        if not policies:
            logger.warning(
                "no_policies_found",
                resource_type=resource_type,
                action=action,
            )
            return PolicyDecision.ABSTAIN

        decisions: List[PolicyDecision] = []
        for policy in policies:
            decision = self._evaluate_policy(policy, user, resource, db)
            decisions.append(decision)
            logger.debug(
                "policy_evaluated",
                policy_type=policy.policy_type.value,
                resource_type=resource_type,
                action=action,
                decision=decision.value,
                user_id=user.user_id,
            )

        if PolicyDecision.ALLOW in decisions:
            return PolicyDecision.ALLOW
        if PolicyDecision.DENY in decisions:
            return PolicyDecision.DENY
        return PolicyDecision.ABSTAIN

    def _evaluate_policy(
        self,
        policy: Policy,
        user: UserContext,
        resource: Any,
        db: Optional[Session],
    ) -> PolicyDecision:
        if policy.custom_evaluator is not None:
            return policy.custom_evaluator(user, resource, db)

        if policy.policy_type == PolicyType.OWNERSHIP:
            return self._check_ownership(user, resource)
        if policy.policy_type == PolicyType.ROLE_ADMIN:
            return PolicyDecision.ALLOW if user.is_admin() else PolicyDecision.DENY
        if policy.policy_type == PolicyType.ROLE_ATTORNEY:
            return PolicyDecision.ALLOW if user.is_attorney() else PolicyDecision.DENY
        if policy.policy_type == PolicyType.ROLE_USER:
            return PolicyDecision.ALLOW if user.role in ("user", "attorney", "admin", "client") else PolicyDecision.DENY
        if policy.policy_type == PolicyType.CASE_COLLABORATOR:
            return self._check_case_collaborator(user, resource, db)

        return PolicyDecision.ABSTAIN

    @staticmethod
    def _check_ownership(user: UserContext, resource: Any) -> PolicyDecision:
        if resource is None:
            return PolicyDecision.ABSTAIN

        owner_id = getattr(resource, "user_id", None)
        if owner_id is None:
            owner_id = getattr(resource, "owner_id", None)

        if owner_id is not None and int(owner_id) == int(user.user_id):
            return PolicyDecision.ALLOW
        return PolicyDecision.DENY

    @staticmethod
    def _check_case_collaborator(
        user: UserContext, resource: Any, db: Optional[Session]
    ) -> PolicyDecision:
        if resource is None or db is None:
            return PolicyDecision.ABSTAIN

        case_id = getattr(resource, "case_id", None)
        if case_id is None:
            case_id = getattr(resource, "id", None)

        if case_id is None:
            return PolicyDecision.ABSTAIN

        try:
            from db.models.cases import CasePresence
            presence = (
                db.query(CasePresence)
                .filter(
                    CasePresence.case_id == int(case_id),
                    CasePresence.user_id == int(user.user_id),
                )
                .first()
            )
            if presence is not None:
                return PolicyDecision.ALLOW
        except Exception:
            logger.exception(
                "collaborator_check_failed",
                case_id=case_id,
                user_id=user.user_id,
            )

        return PolicyDecision.DENY


default_engine = PolicyEngine()


def evaluate(
    user: UserContext,
    resource_type: str,
    action: str,
    resource: Any = None,
    db: Optional[Session] = None,
    engine: Optional[PolicyEngine] = None,
) -> PolicyDecision:
    return (engine or default_engine).evaluate(user, resource_type, action, resource, db)


def _register_builtin_policies():
    register_policy(Policy(PolicyType.OWNERSHIP, "case", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "case", "update"))
    register_policy(Policy(PolicyType.OWNERSHIP, "case", "delete"))
    register_policy(Policy(PolicyType.OWNERSHIP, "case", "upload_document"))
    register_policy(Policy(PolicyType.OWNERSHIP, "case", "add_comment"))
    register_policy(Policy(PolicyType.OWNERSHIP, "case", "add_deadline"))
    register_policy(Policy(PolicyType.CASE_COLLABORATOR, "case", "view"))
    register_policy(Policy(PolicyType.CASE_COLLABORATOR, "case", "add_comment"))
    register_policy(Policy(PolicyType.ROLE_ADMIN, "case", "view"))
    register_policy(Policy(PolicyType.ROLE_ADMIN, "case", "update"))
    register_policy(Policy(PolicyType.ROLE_ADMIN, "case", "delete"))

    register_policy(Policy(PolicyType.OWNERSHIP, "document", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "document", "delete"))

    register_policy(Policy(PolicyType.OWNERSHIP, "deadline", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "deadline", "update"))
    register_policy(Policy(PolicyType.OWNERSHIP, "deadline", "delete"))
    register_policy(Policy(PolicyType.OWNERSHIP, "deadline", "mark_complete"))

    register_policy(Policy(PolicyType.OWNERSHIP, "attachment", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "attachment", "delete"))

    register_policy(Policy(PolicyType.OWNERSHIP, "notification_preference", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "notification_preference", "update"))

    register_policy(Policy(PolicyType.OWNERSHIP, "analysis_job", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "analysis_job", "cancel"))

    register_policy(Policy(PolicyType.OWNERSHIP, "report", "view"))
    register_policy(Policy(PolicyType.OWNERSHIP, "report", "delete"))

    register_policy(Policy(PolicyType.ROLE_ADMIN, "system", "audit_view"))
    register_policy(Policy(PolicyType.ROLE_ADMIN, "system", "user_manage"))
    register_policy(Policy(PolicyType.ROLE_ADMIN, "system", "retention_enforce"))


_register_builtin_policies()