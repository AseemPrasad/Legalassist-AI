"""
Authentication system for LegalAssist AI.
Email-based OTP authentication with JWT session management.
"""

import os
import hashlib
import secrets
import time
import re
from routes import PAGE_LOGIN
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Any
import logging
from config import Config
from db.crud.audit import record_audit_event

import uuid
import jwt

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail
except ImportError:
    sendgrid = None
    Mail = None

from database import (
    SessionLocal,
    get_user_by_email,
    create_user,
    create_otp_verification,
    get_pending_otp,
    mark_otp_as_used,
    cleanup_expired_otps,
    update_user_last_login,
    record_otp_failed_attempt,
    reset_otp_failed_attempts,
    revoke_token,
    is_token_revoked,
    cleanup_expired_revoked_tokens,
    OTPVerification,
    _get_otp_rate_limit_script,
    _otp_rate_limit_key,
    User,
)

logger = logging.getLogger(__name__)

def _is_debug_or_testing_mode() -> bool:
    """Return True when explicit debug/testing flags are enabled."""
    return Config.DEBUG or Config.TESTING


def _is_development_mode() -> bool:
    """Return True when app is running in development-like mode."""
    return Config.is_development()


def _get_jwt_secrets_to_try() -> list[str]:
    secrets_to_try = Config.get_jwt_secrets()
    if not secrets_to_try:
        raise RuntimeError("JWT_SECRET is not configured")
    return secrets_to_try


# Configuration
JWT_SECRET = Config.get_jwt_secret()
JWT_ALGORITHM = Config.JWT_ALGORITHM
JWT_EXPIRY_HOURS = Config.JWT_EXPIRY_HOURS

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

OTP_EXPIRY_MINUTES = Config.OTP_EXPIRY_MINUTES

# OTP Verification Security - Failed Attempt Lockout
OTP_MAX_FAILED_ATTEMPTS = int(os.getenv("OTP_MAX_FAILED_ATTEMPTS", "5"))  # Max failed verification attempts
OTP_LOCKOUT_MINUTES = int(os.getenv("OTP_LOCKOUT_MINUTES", "15"))  # Lockout duration after max attempts
OTP_REQUEST_RATE_LIMIT_MAX = int(os.getenv("OTP_REQUEST_RATE_LIMIT_MAX", str(Config.OTP_REQUEST_RATE_LIMIT_MAX)))
OTP_REQUEST_RATE_LIMIT_HOURS = int(os.getenv("OTP_REQUEST_RATE_LIMIT_HOURS", str(Config.OTP_REQUEST_RATE_LIMIT_HOURS)))


def _hash_otp(otp: str) -> str:
    """Hash OTP code before storing"""
    return hashlib.sha256(otp.encode()).hexdigest()


def _verify_otp_hash(otp: str, otp_hash: str) -> bool:
    """Verify OTP against stored hash"""
    return _hash_otp(otp) == otp_hash


def _otp_rate_limit_keys(email: str, requester_ip: Optional[str] = None) -> list[str]:
    keys = [f"email:{str(email).strip().lower()}"]
    if requester_ip:
        keys.append(f"ip:{str(requester_ip).strip().lower()}")
    return keys


def _reserve_otp_request_slot(identifier: str, window_hours: int, max_requests: int) -> int:
    normalized_identifier = str(identifier).strip().lower()
    if not normalized_identifier:
        raise ValueError("OTP request identifier is required")

    now = datetime.now(timezone.utc)
    rate_limit_start = now - timedelta(hours=window_hours)

    db = SessionLocal()
    try:
        recent_otps = db.query(OTPVerification).filter(
            OTPVerification.email == normalized_identifier,
            OTPVerification.created_at >= rate_limit_start,
        ).count()

        if recent_otps >= max_requests:
            raise ValueError("Too many OTP requests. Please try again later.")

        script = _get_otp_rate_limit_script()
        current = int(script(keys=[_otp_rate_limit_key(normalized_identifier)], args=[window_hours * 60 * 60]))
        if current > max_requests:
            raise ValueError("Too many OTP requests. Please try again later.")
        return current
    finally:
        db.close()


def generate_otp() -> str:
    """Generate a 6-digit OTP code"""
    return f"{secrets.randbelow(1000000):06d}"


def send_otp_email(email: str, otp: str) -> bool:
    """
    Send OTP code via email using SendGrid.
    Returns True if email was sent successfully.
    """
    try:
        api_key = os.getenv("SENDGRID_API_KEY")
        from_email = os.getenv("SENDGRID_FROM_EMAIL", "noreply@legalassist.ai")

        if not api_key or sendgrid is None:
            if _is_debug_or_testing_mode():
                logger.warning("SendGrid API key not configured or sendgrid package not installed - using masked OTP logging for debug/test mode")
                logger.debug(f"OTP for {email}: [MASKED]")
                return True  # Simulate success only in explicit debug/testing environments
            logger.error(
                f"SendGrid API key not configured or sendgrid package not installed — OTP delivery failed for {email}. "
                "Set SENDGRID_API_KEY and install requirements-notifications.txt to enable email authentication."
            )
            return False

        sg = sendgrid.SendGridAPIClient(api_key=api_key)

        subject = "Your LegalAssist AI Login OTP"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #2d2dff;">LegalAssist AI Login</h2>
            <p>Your One-Time Password (OTP) for login is:</p>
            <h1 style="background-color: #f0f0f0; padding: 20px; text-align: center; letter-spacing: 5px; font-size: 32px;">
                {otp}
            </h1>
            <p>This OTP will expire in {OTP_EXPIRY_MINUTES} minutes.</p>
            <p><strong>Do not share this code with anyone.</strong></p>
            <hr>
            <p style="color: #666; font-size: 12px;">
                If you didn't request this OTP, please ignore this email.
            </p>
        </body>
        </html>
        """

        message = Mail(
            from_email=from_email,
            to_emails=email,
            subject=subject,
            html_content=body,
        )

        response = sg.send(message)
        logger.info(f"OTP email sent to {email}, status code: {response.status_code}")
        return 200 <= response.status_code < 300

    except Exception as e:
        logger.error(f"Failed to send OTP email to {email}: {str(e)}")
        if _is_debug_or_testing_mode():
            logger.debug(f"OTP for {email}: [MASKED]")
        else:
            logger.warning(f"OTP delivery failed for {email} (check email service config)")
        return False


def request_otp(email: str, requester_ip: Optional[str] = None) -> Tuple[bool, str]:
    """
    Request OTP for email authentication.
    Returns (success, message).
    """
    # Validate email format
    if not email or not EMAIL_REGEX.match(email):
        return False, "Invalid email address"

    db = SessionLocal()
    try:
        # Generate OTP
        otp = generate_otp()
        otp_hash = _hash_otp(otp)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)

        # Store OTP
        try:
            create_otp_verification(
                db,
                email,
                otp_hash,
                expires_at,
                max_requests_per_hour=OTP_REQUEST_RATE_LIMIT_MAX,
                requester_ip=requester_ip,
            )
        except ValueError as exc:
            return False, str(exc)

        # Send OTP email
        email_sent = send_otp_email(email, otp)

        if email_sent:
            # Create user if doesn't exist
            user = get_user_by_email(db, email)
            if not user:
                create_user(db, email)
                logger.info(f"New user created: {email}")

            return True, "OTP sent to your email"
        else:
            return False, "Failed to send OTP email. Please try again."

    except Exception as e:
        logger.error(f"Error requesting OTP for {email}: {str(e)}")
        return False, f"Error: {str(e)}"
    finally:
        db.close()


def verify_otp_and_create_token(email: str, otp: str) -> Tuple[bool, str, Optional[str]]:
    """
    Verify OTP and create JWT token with brute-force protection.
    Returns (success, message, token).
    
    Security features:
    - Track failed verification attempts per OTP
    - Lock OTP after max failed attempts
    - Require user to request a new OTP after lockout
    """
    db = SessionLocal()
    try:
        # Get pending OTP
        otp_record = get_pending_otp(db, email)

        if not otp_record:
            return False, "OTP expired or not found. Please request a new one.", None

        # Check if OTP is locked due to too many failed attempts
        if otp_record.is_locked():
            # Safely normalize naive datetimes to UTC
            locked_until = otp_record.locked_until
            if locked_until and locked_until.tzinfo is None:
                # Treat naive datetime as local server time and convert to UTC
                locked_until = locked_until.astimezone(timezone.utc)
            
            remaining_time = (locked_until - datetime.now(timezone.utc)).total_seconds() / 60
            logger.warning(f"OTP verification attempt for {email} blocked - OTP is locked (remaining time: {remaining_time:.1f} minutes)")
            return False, f"Too many failed attempts. Please request a new OTP after {int(remaining_time)} minutes.", None

        # Verify OTP
        if not _verify_otp_hash(otp, otp_record.otp_hash):
            # Record failed attempt and check if lockout is needed
            record_otp_failed_attempt(
                db, 
                otp_record.id, 
                lockout_duration_minutes=OTP_LOCKOUT_MINUTES,
                max_failed_attempts=OTP_MAX_FAILED_ATTEMPTS
            )
            
            # Check if OTP is now locked after this attempt
            db.refresh(otp_record)
            if otp_record.is_locked():
                logger.warning(
                    f"OTP for {email} locked after {otp_record.failed_attempts} failed verification attempts"
                )
                return False, f"Too many failed attempts (limit: {OTP_MAX_FAILED_ATTEMPTS}). OTP is now locked. Please request a new OTP.", None
            
            attempts_remaining = OTP_MAX_FAILED_ATTEMPTS - otp_record.failed_attempts
            logger.info(f"Failed OTP verification for {email}. Attempts remaining: {attempts_remaining}")
            return False, f"Invalid OTP code. {attempts_remaining} attempts remaining before lockout.", None

        # OTP is valid - reset failed attempts and mark as used
        reset_otp_failed_attempts(db, otp_record.id)
        mark_otp_as_used(db, otp_record.id)

        # Get or create user
        user = get_user_by_email(db, email)
        if not user:
            user = create_user(db, email)

        # Update last login
        update_user_last_login(db, user.id)

        # Create JWT token
        token = create_jwt_token(user.id, user.email)

        record_audit_event(
            db,
            actor=f"user:{user.id}",
            actor_user_id=user.id,
            action="login_success",
            resource="auth:session",
            metadata={"email_domain": user.email.split("@")[-1] if "@" in user.email else None},
        )

        logger.info(f"User logged in successfully: {email} (user_id={user.id})")
        return True, "Login successful", token

    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {str(e)}")
        return False, f"Error: {str(e)}", None
    finally:
        db.close()


# =========================================================================
# JWT AUTHENTICATION CONSTANTS & CONFIGURATION
# =========================================================================
# The following constants define the strict claims required for our JSON Web Tokens (JWT).
# 
# What are Issuer (iss) and Audience (aud) claims?
# ------------------------------------------------
# - Issuer (iss): Identifies the principal that issued the JWT. In a distributed 
#   system, this prevents tokens issued by one service (e.g., an internal billing API) 
#   from being used in another service (e.g., this user-facing application).
# - Audience (aud): Identifies the recipients that the JWT is intended for. Each
#   service validating the token must verify that it is listed as an intended audience.
# 
# Why This Matters (Security Justification):
# ------------------------------------------
# Without these checks, an attacker could potentially take a token validly issued 
# by a different but related system (using the same shared secret or public key) 
# and use it here. This vulnerability is known as "Cross-JWT Confusion" or 
# "Token Substitution". By strictly enforcing `iss` and `aud`, we cryptographically 
# guarantee that the token was explicitly generated *by* LegalAssist AI and 
# *for* LegalAssist AI users, hardening our API security against unauthorized 
# or external token usage.
# =========================================================================

JWT_ISSUER = os.getenv("JWT_ISSUER", "legalassist.ai")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "legalassist-users")


def create_jwt_token(user_id: int, email: str) -> str:
    """
    Create a highly secure JWT token for an authenticated user.
    
    This function generates a JSON Web Token containing essential claims
    used to verify the user's identity and session validity. It includes
    both standard registered claims (like exp, iat, iss, aud) and 
    custom private claims (like user_id, email, type).
    
    Parameters:
    -----------
    user_id : int
        The primary key ID of the user in the database.
    email : str
        The user's registered email address.
        
    Returns:
    --------
    str
        A fully encoded and cryptographically signed JWT string.
    """
    
    # Delegate JWT creation to the canonical API auth implementation
    from api.auth import create_access_token

    data = {"user_id": user_id, "email": email}
    return create_access_token(data)


def verify_jwt_token(token: str) -> Optional[dict]:
    """Delegate verification to the canonical API auth implementation.

    Returns the payload dict on success or None on failure.
    """
    from api.auth import verify_token
    try:
        return verify_token(token)
    except Exception:
        return None


def revoke_jwt_token(token: str) -> bool:
    """
    Revokes a JWT token so it can no longer be used.
    
    This function is primarily used during the user logout process to 
    immediately invalidate the active session. Because JWTs are stateless 
    by default, we implement a stateful "blacklist" using the token's JTI.
    
    Parameters:
    -----------
    token : str
        The raw JWT string to be revoked.
        
    Returns:
    --------
    bool
        True if the token was successfully added to the revocation list,
        False if the operation failed or the token was invalid.
    """
    # Delegate revocation to the API auth implementation
    from api.auth import revoke_jwt_token as _api_revoke
    try:
        return _api_revoke(token)
    except Exception:
        return False


def get_current_user_from_token(token: str) -> Optional[User]:
    """
    Retrieve the full User database model object from a given JWT token.
    
    This acts as a convenience wrapper around verify_jwt_token() for 
    endpoints or functions that need the actual ORM object rather than 
    just the raw claims dictionary.
    
    Parameters:
    -----------
    token : str
        The raw JWT access token.
        
    Returns:
    --------
    Optional[User]
        The User ORM model if the token is valid and the user exists.
        Returns None otherwise.
    """
    # Map to underlying API auth verification, then return ORM user if present
    payload = verify_jwt_token(token)
    if not payload:
        return None

    email = payload.get("email")
    if not email:
        return None

    db = SessionLocal()
    try:
        return get_user_by_email(db, email)
    finally:
        db.close()


def cleanup_old_data() -> int:
    """
    Cleanup expired OTPs and expired revoked tokens.
    Returns count of cleaned up records.
    """
    db = SessionLocal()
    try:
        deleted_otps = cleanup_expired_otps(db)
        deleted_tokens = cleanup_expired_revoked_tokens(db)
        total_deleted = deleted_otps + deleted_tokens
        logger.info(f"Cleaned up {deleted_otps} expired OTPs and {deleted_tokens} expired revoked tokens")
        return total_deleted
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return 0
    finally:
        db.close()


# ==================== Streamlit Session Helpers ====================


def init_auth_session():
    """Initialize authentication state in Streamlit session"""
    import streamlit as st
    
    # Initialize auth state keys
    if "user_token" not in st.session_state:
        st.session_state.user_token = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "session_created_at" not in st.session_state:
        st.session_state.session_created_at = None


def validate_auth_state() -> bool:
    """
    Validate authentication state with multi-tab resilience.
    Returns True if valid, False if session needs reset.
    """
    import streamlit as st
    from datetime import datetime, timezone
    
    if not st.session_state.get("is_authenticated"):
        return False
    
    token = st.session_state.get("user_token")
    if not token:
        return False
    
    # Verify token is still valid
    try:
        payload = verify_jwt_token(token)
        if not payload:
            # Token invalid/expired - clear state
            clear_auth_session()
            return False
        
        # Check for forced logout flag (set by logout in any tab)
        if st.session_state.get("force_logout"):
            clear_auth_session()
            st.session_state.force_logout = False
            return False
        
        return True
    except Exception:
        clear_auth_session()
        return False


def clear_auth_session():
    """Clear all authentication state"""
    import streamlit as st
    
    st.session_state.user_token = None
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.is_authenticated = False
    st.session_state.session_created_at = None


def force_logout_all_tabs():
    """Force logout across all tabs by setting flag"""
    import streamlit as st
    
    st.session_state.force_logout = True
    st.session_state.user_token = None
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.is_authenticated = False


def login_user(email: str) -> bool:
    """
    Initiate login by sending OTP.
    Stores email in session for verification step.
    """
    import streamlit as st

    init_auth_session()
    st.session_state.pending_email = email

    success, message = request_otp(email)
    if success:
        st.session_state.otp_sent = True
        st.session_state.pending_email = email
    return success


def verify_login(otp: str) -> bool:
    """
    Verify OTP and complete login.
    Returns True if login successful.
    """
    import streamlit as st

    init_auth_session()
    email = st.session_state.get("pending_email")

    if not email:
        return False

    success, message, token = verify_otp_and_create_token(email, otp)

    if success and token:
        st.session_state.user_token = token
        st.session_state.user_email = email

        # Get user ID from token payload
        payload = verify_jwt_token(token)
        if payload:
            st.session_state.user_id = payload.get("user_id")

        st.session_state.is_authenticated = True
        st.session_state.pending_email = None
        st.session_state.otp_sent = False

        return True

    return False


def logout_user():
    """
    Logout current user, revoke their JWT token, and aggressively clear the session state.
    
    This function is the authoritative source for user termination in the application.
    It implements a "Scorched Earth" policy for session data to guarantee that NO 
    personally identifiable information (PII) or authentication artifacts remain
    in the browser's memory after the user clicks 'Logout'.
    
    SECURITY RATIONALE:
    ------------------
    1. SHARED TERMINALS: In legal environments, users may share workstations. 
       If session data (like case IDs or document summaries) is not purged, 
       the next user could potentially view the previous user's sensitive data.
    
    2. STALE STATE BUGS: Streamlit's reactive model sometimes retains values
       for widgets that are no longer visible. Explicitly deleting keys
       from st.session_state forces a clean reset.
    
    3. REPLAY PROTECTION: By revoking the token in the database, we ensure 
       the session is dead on the server side as well as the client side.
    """
    import streamlit as st

    logger.info("Performing global logout and session state purge...")
    
    # Ensure session state is initialized before we start clearing it
    init_auth_session()
    
    # Step 1: Revoke the token if it exists in the database.
    # This prevents the token from being used in any subsequent API calls
    # even if it is intercepted from the client's network traffic.
    token = st.session_state.get("user_token")
    if token:
        try:
            revoke_jwt_token(token)
            logger.debug("Active JWT token revoked in database.")
        except Exception as e:
            logger.error(f"Failed to revoke token during logout: {str(e)}")
            # We continue with session clearing even if revocation fails
            # to prioritize local data privacy.
    
    # Step 2: Set force logout flag for multi-tab synchronization
    force_logout_all_tabs()
    
    # Step 3: Aggressive Session State Wipe.
    # Instead of just setting individual keys to None, we iterate through
    # every key currently registered in the Streamlit session and delete it.
    # This ensures that ANY data stored by the app (including custom keys
    # added by individual pages) is completely erased.
    
    # We use list() to create a copy of the keys to avoid "RuntimeError: 
    # dictionary changed size during iteration".
    all_keys = list(st.session_state.keys())
    
    for key in all_keys:
        try:
            del st.session_state[key]
        except KeyError:
            # Handle potential race conditions where a key might have 
            # been removed by another process/thread (unlikely but safe).
            pass
            
    logger.info(f"Successfully cleared {len(all_keys)} session state keys.")
    
    # NOTE: The caller (e.g., app.py) is responsible for calling st.rerun()
    # to restart the UI flow after this function returns.


def require_auth() -> bool:
    """
    Check if user is authenticated.
    Use this in pages that require login.
    Returns True if authenticated, False otherwise.
    """
    import streamlit as st

    init_auth_session()
    
    # Use validation with multi-tab support
    if validate_auth_state():
        return True
    
    # Token invalid/expired - clear state
    clear_auth_session()
    return False



def redirect_to_login():
    """Redirect to login page"""
    import streamlit as st

    st.switch_page(PAGE_LOGIN)


def get_current_user_id() -> Optional[int]:
    """Get current user ID from session"""
    import streamlit as st

    init_auth_session()

    if st.session_state.is_authenticated and st.session_state.user_id:
        return st.session_state.user_id

    return None


def get_current_user_email() -> Optional[str]:
    """Get current user email from session"""
    import streamlit as st

    init_auth_session()

    if st.session_state.is_authenticated and st.session_state.user_email:
        return st.session_state.user_email

    return None
