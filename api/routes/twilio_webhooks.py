"""
Twilio SMS Status Webhook Endpoint
POST /api/v1/webhooks/twilio/sms-status - Receive Twilio delivery status callbacks

Graceful-degradation design
-----------------------------
Twilio expects HTTP 2xx from webhook endpoints. A non-2xx response triggers
Twilio's retry logic, which can generate a *retry storm* — repeated deliveries
that increase load and obscure the underlying operational issue.

This module prevents retry storms by:

1. **Missing RequestValidator dependency** — if the ``twilio`` package (or its
   ``request_validator`` sub-module) is not installed, returning HTTP 503 would
   cause Twilio to retry indefinitely.  Instead we return HTTP 200 + empty TwiML
   and emit a CRITICAL-level structured log so operators get an actionable alert.

2. **Missing Twilio auth token** — same treatment: CRITICAL log + HTTP 200.

3. **Invalid signature** — the request is definitely not from Twilio; we return
   HTTP 403 (Forbidden).  Twilio does NOT retry 403 responses.

4. **Valid request** — delivery status is persisted and HTTP 200 is returned.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import structlog
from fastapi import APIRouter, Form, Request, Response, status

from config import Config
from database import (
    NotificationLog,
    NotificationStatus,
    SessionLocal,
)

router = APIRouter(prefix="/api/v1/webhooks/twilio", tags=["webhooks"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency — resolved once at import time
# ---------------------------------------------------------------------------
# We attempt the import here so the availability check inside the endpoint is
# a simple boolean test (O(1)) rather than a repeated import round-trip.

try:
    from twilio.request_validator import RequestValidator as _TwilioRequestValidator
    _VALIDATOR_AVAILABLE = True
except ImportError:
    _TwilioRequestValidator = None  # type: ignore[assignment, misc]
    _VALIDATOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Twilio delivery status → internal NotificationStatus mapping
# ---------------------------------------------------------------------------

_TWILIO_STATUS_MAP = {
    "delivered":   NotificationStatus.SENT,
    "failed":      NotificationStatus.FAILED,
    "undelivered": NotificationStatus.FAILED,
    "bounced":     NotificationStatus.BOUNCED,
}


# ---------------------------------------------------------------------------
# TwiML helpers
# ---------------------------------------------------------------------------

_EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
_TWIML_MEDIA_TYPE = "text/xml"


def _twiml_ok() -> Response:
    """HTTP 200 with an empty TwiML body — tells Twilio the request was received."""
    return Response(content=_EMPTY_TWIML, media_type=_TWIML_MEDIA_TYPE)


def _forbidden() -> Response:
    """HTTP 403 — Twilio does NOT retry 403 responses, so this is safe."""
    return Response(status_code=status.HTTP_403_FORBIDDEN)


# ---------------------------------------------------------------------------
# Status persistence — isolated transaction
# ---------------------------------------------------------------------------

def _persist_status_update(message_sid: str, new_status: NotificationStatus) -> None:
    """Update the NotificationLog row identified by *message_sid*.

    Opens its own ``SessionLocal`` session so a DB failure only affects this
    call and does not propagate to the HTTP response (which must always be
    200 to avoid Twilio retries).
    """
    db = SessionLocal()
    try:
        log_entry = (
            db.query(NotificationLog)
            .filter(NotificationLog.message_id == message_sid)
            .first()
        )

        if log_entry is None:
            logger.info(
                "twilio_webhook_no_matching_log",
                message_sid=message_sid,
                new_status=new_status.value,
            )
            return

        log_entry.status = new_status
        if new_status == NotificationStatus.SENT:
            log_entry.delivered_at = dt.datetime.now(dt.timezone.utc)

        db.commit()
        logger.info(
            "twilio_webhook_status_persisted",
            message_sid=message_sid,
            new_status=new_status.value,
            notification_log_id=log_entry.id,
        )

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error(
            "twilio_webhook_persist_failed",
            message_sid=message_sid,
            error=str(exc),
            exc_info=True,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/sms-status",
    status_code=status.HTTP_200_OK,
    summary="Receive Twilio SMS delivery status callbacks",
)
async def twilio_sms_status_webhook(
    request: Request,
    MessageSid: Optional[str] = Form(default=None),
    MessageStatus: Optional[str] = Form(default=None),
    SmsSid: Optional[str] = Form(default=None),
    SmsStatus: Optional[str] = Form(default=None),
) -> Response:
    """Handle Twilio SMS delivery status callbacks with graceful degradation.

    Step 1 — Dependency check
    --------------------------
    If ``twilio.request_validator.RequestValidator`` is unavailable we cannot
    validate signatures.  Returning 503 would trigger Twilio's retry logic and
    create a retry storm.  Instead we:
      - Emit a CRITICAL log with actionable remediation instructions.
      - Return HTTP 200 + empty TwiML to acknowledge receipt.

    Step 2 — Auth token check
    --------------------------
    Same treatment if ``TWILIO_AUTH_TOKEN`` is not set in the environment.

    Step 3 — Signature validation
    ------------------------------
    Uses ``RequestValidator.validate()`` against the posted form fields and the
    ``X-Twilio-Signature`` header.  An invalid signature means the request is
    NOT from Twilio; we return HTTP 403.  Twilio does not retry 403 responses,
    so this path is safe.

    Step 4 — Status persistence
    ----------------------------
    Delivery status is written to ``NotificationLog`` in an isolated DB
    transaction.  Any DB failure is logged but does not change the HTTP 200
    response (preventing a spurious Twilio retry).
    """

    # ------------------------------------------------------------------
    # Step 1: Validator dependency check
    # ------------------------------------------------------------------
    if not _VALIDATOR_AVAILABLE:
        logger.critical(
            "twilio_webhook_validator_unavailable",
            remedy=(
                "The 'twilio' package is not installed or "
                "twilio.request_validator cannot be imported. "
                "Run: pip install twilio  — then redeploy the service. "
                "Returning HTTP 200 to prevent a Twilio retry storm."
            ),
        )
        return _twiml_ok()

    # ------------------------------------------------------------------
    # Step 2: Auth token check
    # ------------------------------------------------------------------
    auth_token = Config.get_twilio_auth_token()
    if not auth_token:
        logger.critical(
            "twilio_webhook_auth_token_missing",
            remedy=(
                "TWILIO_AUTH_TOKEN environment variable is not set. "
                "Webhook signatures cannot be validated. "
                "Set TWILIO_AUTH_TOKEN immediately. "
                "Returning HTTP 200 to prevent a Twilio retry storm."
            ),
        )
        return _twiml_ok()

    # ------------------------------------------------------------------
    # Step 3: Signature validation
    # ------------------------------------------------------------------
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)

    try:
        # ``request.form()`` must be awaited; Starlette caches the result so
        # this is safe even though FastAPI already parsed the Form fields.
        form_fields = dict(await request.form())
        validator = _TwilioRequestValidator(auth_token)
        is_valid = validator.validate(url, form_fields, signature)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "twilio_webhook_validation_error",
            error=str(exc),
            exc_info=True,
        )
        # Return 403 (not retried) rather than 500 (retried).
        return _forbidden()

    if not is_valid:
        logger.warning(
            "twilio_webhook_invalid_signature",
            url=url,
            signature_present=bool(signature),
        )
        return _forbidden()

    # ------------------------------------------------------------------
    # Step 4: Process delivery status update
    # ------------------------------------------------------------------
    message_sid = MessageSid or SmsSid
    raw_status = (MessageStatus or SmsStatus or "").lower().strip()

    if not message_sid:
        logger.warning(
            "twilio_webhook_missing_message_sid",
            raw_status=raw_status,
        )
        return _twiml_ok()

    new_status = _TWILIO_STATUS_MAP.get(raw_status)
    if new_status is None:
        # Informational statuses (queued, sending, accepted) — nothing to persist.
        logger.debug(
            "twilio_webhook_untracked_status",
            message_sid=message_sid,
            raw_status=raw_status,
        )
        return _twiml_ok()

    _persist_status_update(message_sid, new_status)
    return _twiml_ok()
