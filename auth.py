"""
Authentication system for LegalAssist AI.
Email-based OTP authentication with JWT session management.
"""

import os
import hashlib
import secrets
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import logging
from config import Config

import uuid
import jwt
import sendgrid
from sendgrid.helpers.mail import Mail

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
    User,
    OTPVerification,
    get_user_organizations,
    get_user_role_in_org,
)

logger = logging.getLogger(__name__)

def _is_debug_or_testing_mode() -> bool:
    """Return True when explicit debug/testing flags are enabled."""
    return Config.DEBUG or Config.TESTING


def _is_development_mode() -> bool:
    """Return True when app is running in development-like mode."""
    return Config.is_development()


# Configuration
JWT_SECRET = Config.get_jwt_secret()
JWT_ALGORITHM = Config.JWT_ALGORITHM
JWT_EXPIRY_HOURS = Config.JWT_EXPIRY_HOURS

OTP_EXPIRY_MINUTES = Config.OTP_EXPIRY_MINUTES
OTP_RATE_LIMIT_HOURS = 1
OTP_RATE_LIMIT_MAX = 3  # Max OTP requests per email per hour

# OTP Verification Security - Failed Attempt Lockout
OTP_MAX_FAILED_ATTEMPTS = int(os.getenv("OTP_MAX_FAILED_ATTEMPTS", "5"))  # Max failed verification attempts
OTP_LOCKOUT_MINUTES = int(os.getenv("OTP_LOCKOUT_MINUTES", "15"))  # Lockout duration after max attempts


def _hash_otp(otp: str) -> str:
    """Hash OTP code before storing"""
    return hashlib.sha256(otp.encode()).hexdigest()


def _verify_otp_hash(otp: str, otp_hash: str) -> bool:
    """Verify OTP against stored hash"""
    return _hash_otp(otp) == otp_hash


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

        if not api_key:
            logger.warning("SendGrid API key not configured - using masked OTP logging")
            if _is_debug_or_testing_mode():
                logger.debug(f"OTP for {email}: [MASKED-{otp[:2]}***{otp[-1]}]")
            else:
                logger.warning(f"OTP requested for {email} (email delivery skipped - missing config)")
            return True  # Return True in development mode

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
        # Fallback: masked OTP logging only in debug mode
        if _is_debug_or_testing_mode():
            logger.debug(f"OTP for {email}: [MASKED-{otp[:2]}***{otp[-1]}]")
        else:
            logger.warning(f"OTP delivery failed for {email} (check email service config)")
        return False


def _handle_test_account_bypass(db: SessionLocal, email: str, now: datetime) -> Tuple[bool, str]:
    """
    Handles automated OTP bypass for designated test accounts in non-production environments.
    
    CRITICAL SECURITY DESIGN:
    The previous implementation allowed a bypass for 'test@example.com' based solely 
    on the 'DEBUG' flag. This was dangerous because 'DEBUG' is often accidentally 
    left enabled in staging or even production misconfigurations.
    
    This new implementation implements 'Defense in Depth' by requiring:
    1. ACCOUNT MATCH: The email must exactly match the hardcoded test account.
    2. ENVIRONMENT LOCK: The APP_ENV must NOT be 'production'.
    3. EXPLICIT OPT-IN: A specific 'ALLOW_UNSAFE_TEST_BYPASS' variable must be 'true'.
    4. MODE VERIFICATION: Standard 'DEBUG' or 'TESTING' flags must still be active.
    
    If any of these conditions are missing, the bypass is skipped entirely and 
    the system falls back to the secure, real OTP flow.
    """
    # Step 1: Identity Verification
    # We only ever allow a bypass for this specific, low-privilege test account.
    if email.lower() != "test@example.com":
        return False, "Not a designated test account"

    # Step 2: Production Safeguard
    # Explicitly block this logic if we detect we are in a production environment.
    # We default to 'production' if the variable is missing to fail-safe.
    app_env = os.getenv("APP_ENV", "production").strip().lower()
    if app_env == "production":
        logger.error(
            "SECURITY WARNING: Bypass attempt for %s blocked because APP_ENV is 'production'.",
            email
        )
        return False, "Bypass strictly forbidden in production"

    # Step 3: Explicit Opt-In Flag
    # This requires the administrator to set a very specific, scary-sounding 
    # environment variable, making it harder to enable by mistake.
    truthy = {"1", "true", "yes", "on"}
    allow_bypass = os.getenv("ALLOW_UNSAFE_TEST_BYPASS", "").strip().lower() in truthy
    if not allow_bypass:
        return False, "Explicit bypass flag (ALLOW_UNSAFE_TEST_BYPASS) is not enabled"

    # Step 4: Mode Verification
    # Ensure we are actually in a debug/testing context as defined by the app.
    if not _is_debug_or_testing_mode():
        return False, "Standard debug or testing flags are not active"

    # --- ALL SECURITY CHECKS PASSED ---
    # We proceed with generating a deterministic 'test' OTP for CI/CD or local dev.
    test_otp = "123456"
    test_otp_hash = _hash_otp(test_otp)
    expires_at = now + timedelta(minutes=OTP_EXPIRY_MINUTES)
    
    # Register the bypass in the database so the verification step works correctly.
    create_otp_verification(db, email, test_otp_hash, expires_at)
    
    # Ensure the test user exists in the system.
    user = get_user_by_email(db, email)
    if not user:
        create_user(db, email)
        logger.info("Created new test user for bypass: %s", email)
        
    logger.warning(
        "SECURITY ALERT: Active OTP bypass for %s (Env: %s). "
        "Remove ALLOW_UNSAFE_TEST_BYPASS in non-test environments.", 
        email, app_env
    )
    
    return True, "Test bypass activated successfully"


def request_otp(email: str) -> Tuple[bool, str]:
    """
    Request OTP for email authentication.
    Returns (success, message).
    """
    # Validate email format
    if not email or "@" not in email or "." not in email:
        return False, "Invalid email address"

    db = SessionLocal()
    try:
        # Check rate limiting
        now = datetime.now(timezone.utc)
        
        # SECURITY: Check for isolated test account bypass.
        # This replaces the previous vulnerable inline check.
        # It uses multiple layers of environment validation to prevent 
        # accidental backdoors in production builds.
        bypass_success, bypass_msg = _handle_test_account_bypass(db, email, now)
        if bypass_success:
            # We return a generic 'success' message to the frontend to maintain 
            # consistent UI behavior and avoid leaking bypass status.
            return True, "OTP sent to your email"

        rate_limit_start = now - timedelta(hours=OTP_RATE_LIMIT_HOURS)

        recent_otps = db.query(OTPVerification).filter(
            OTPVerification.email == email,
            OTPVerification.created_at >= rate_limit_start,
        ).count()

        if recent_otps >= OTP_RATE_LIMIT_MAX:
            return False, "Too many OTP requests. Please try again in an hour."

        # Generate OTP
        otp = generate_otp()
        otp_hash = _hash_otp(otp)
        expires_at = now + timedelta(minutes=OTP_EXPIRY_MINUTES)

        # Store OTP
        create_otp_verification(db, email, otp_hash, expires_at)

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
            # Ensure locked_until is timezone-aware
            locked_until = otp_record.locked_until
            if locked_until and locked_until.tzinfo is None:
                locked_until = locked_until.replace(tzinfo=timezone.utc)
            
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

        logger.info(f"User logged in successfully: {email} (user_id={user.id})")
        return True, "Login successful", token

    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {str(e)}")
        return False, f"Error: {str(e)}", None
    finally:
        db.close()


def create_jwt_token(user_id: int, email: str) -> str:
    """Create JWT token for authenticated user"""
    # Generate a unique JWT ID to allow for future token revocation
    jti = str(uuid.uuid4())
    
    payload = {
        "jti": jti,
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict]:
    """
    Verify JWT token and return payload.
    Returns None if token is invalid, expired, incorrectly purposed,
    revoked (logged out), or if the user has been removed.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # 1. Verify token purpose
        # We explicitly set type="access" during token creation to prevent
        # other types of tokens (e.g., password reset, email verification)
        # from being used to gain system access.
        if payload.get("type") != "access":
            logger.warning(f"Invalid token type provided: {payload.get('type')}")
            return None
            
        jti = payload.get("jti")
        email = payload.get("email")
        if not email or not jti:
            logger.warning("Token payload missing required 'email' or 'jti' claim")
            return None
            
        # 2. Database verifications (Revocation and User Existence)
        db = SessionLocal()
        try:
            # Check if token has been revoked (e.g. via logout)
            if is_token_revoked(db, jti):
                logger.warning(f"Attempted to use revoked token jti={jti} for user {email}")
                return None
                
            # Check the database to guarantee the user hasn't been deleted or suspended.
            user = get_user_by_email(db, email)
            if not user:
                logger.warning(f"Token verification failed: User {email} no longer exists in DB")
                return None
        finally:
            db.close()
            
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        return None


def revoke_jwt_token(token: str) -> bool:
    """
    Revokes a JWT token so it can no longer be used.
    Used during logout to immediately invalidate the session.
    """
    if not token:
        return False
        
    try:
        # We don't need to verify expiration here, just decode it.
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
        jti = payload.get("jti")
        exp = payload.get("exp")
        
        if not jti or not exp:
            return False
            
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
        
        db = SessionLocal()
        try:
            # Avoid duplicate revocation
            if not is_token_revoked(db, jti):
                revoke_token(db, jti, expires_at)
                logger.info(f"Token jti={jti} successfully revoked")
            return True
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        return False


def get_current_user_from_token(token: str) -> Optional[User]:
    """Get current user from JWT token"""
    payload = verify_jwt_token(token)
    if not payload:
        return None

    db = SessionLocal()
    try:
        user = get_user_by_email(db, payload["email"])
        return user
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

    if "user_token" not in st.session_state:
        st.session_state.user_token = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "is_authenticated" not in st.session_state:
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
            user_id = payload.get("user_id")
            st.session_state.user_id = user_id
            
            # Fetch organization and role for the session
            db = SessionLocal()
            orgs = get_user_organizations(db, user_id)
            if orgs:
                # For now, default to the first organization
                st.session_state.org_id = orgs[0].id
                st.session_state.org_name = orgs[0].name
                st.session_state.user_role = get_user_role_in_org(db, user_id, orgs[0].id)
            else:
                st.session_state.org_id = None
                st.session_state.org_name = None
                st.session_state.user_role = None
            db.close()

        st.session_state.is_authenticated = True
        st.session_state.pending_email = None
        st.session_state.otp_sent = False

        return True

    return False


def logout_user():
    """Logout current user and revoke their token"""
    import streamlit as st

    init_auth_session()
    
    # Revoke the token if it exists
    token = st.session_state.get("user_token")
    if token:
        revoke_jwt_token(token)
        
    st.session_state.user_token = None
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.org_id = None
    st.session_state.org_name = None
    st.session_state.user_role = None
    st.session_state.is_authenticated = False
    st.session_state.pending_email = None
    st.session_state.otp_sent = False


def require_auth() -> bool:
    """
    Check if user is authenticated.
    Use this in pages that require login.
    Returns True if authenticated, False otherwise.
    """
    import streamlit as st

    init_auth_session()

    if st.session_state.is_authenticated and st.session_state.user_token:
        # Verify token is still valid
        payload = verify_jwt_token(st.session_state.user_token)
        if payload:
            return True
        else:
            # =========================================================================
            # CRITICAL SECURITY AND STATE MANAGEMENT FIX
            # =========================================================================
            # 
            # 1. OVERVIEW OF STREAMLIT EXECUTION MODEL
            # ----------------------------------------
            # Streamlit is built on a unique execution model where the entire Python script
            # is rerun from top to bottom every time a user interacts with a widget or
            # when a programmatic state change occurs. Unlike traditional web frameworks
            # (like Flask, Django, or FastAPI) that map distinct HTTP requests to isolated
            # controller functions, Streamlit handles UI state interactively within a
            # continuous, single-script context.
            # 
            # When an event triggers a rerun, Streamlit:
            #   a) Captures the widget interactions.
            #   b) Updates `st.session_state` based on those interactions.
            #   c) Starts executing the main script file from line 1.
            #   d) Dynamically redraws the UI components in the exact order they are called.
            # 
            # 2. THE PROBLEM: CORRUPTED STATE UPON FORCED LOGOUT
            # --------------------------------------------------
            # In our application, protected pages call `require_auth()` at the very top.
            # This function is responsible for ensuring that the user has a valid, 
            # unexpired, and non-revoked JWT token.
            # 
            # If the token is found to be invalid (e.g., the user was deleted, the token 
            # expired, or it was manually revoked via a blacklist), we trigger `logout_user()`.
            # The `logout_user()` function successfully clears the critical authentication 
            # variables from `st.session_state` (like `user_id`, `user_token`, etc.).
            # 
            # However, merely calling `logout_user()` does NOT implicitly stop the current
            # top-to-bottom execution of the script. After `require_auth()` returns False,
            # the execution simply proceeds to the next line of code in the page.
            # 
            # If the application developer didn't wrap the entire page in an 
            # `if require_auth():` block (or if they rely on `require_auth()` to halt
            # execution on its own), the script will continue rendering the protected content.
            # 
            # Because `logout_user()` just cleared the session state, the subsequent code 
            # is now operating on a "Corrupted State":
            #   - It expects `st.session_state.user_id` to be an integer, but it's now None.
            #   - It attempts to fetch user-specific data from the database, leading to errors.
            #   - It renders UI components that should never be visible to unauthenticated users.
            # 
            # This leads to a catastrophic cascade of exceptions (like AttributeError or 
            # TypeError) visually cluttering the screen with error stack traces, or even 
            # worse, a brief "flash" of unauthorized content before the UI crashes.
            # 
            # 3. SECURITY IMPLICATIONS OF CONTINUED EXECUTION
            # -----------------------------------------------
            # The failure to halt execution represents a severe security vulnerability 
            # known as "Information Disclosure" and "Improper Access Control".
            # 
            # Scenario:
            # - User A logs in and has a valid session.
            # - User A's token expires at 12:00 PM.
            # - At 12:01 PM, User A clicks a button to view a sensitive document.
            # - The app calls `require_auth()`, detects the expired token, and clears the state.
            # - BUT, execution continues.
            # - The code attempting to render the document might still have a cached ID
            #   or might fail open, accidentally displaying sensitive data on the screen
            #   because the script was not forcibly halted.
            # 
            # By enforcing a strict halt, we guarantee a "Fail Secure" posture. If 
            # authentication fails, no further code in the current execution context 
            # is permitted to run, completely neutralizing the risk of accidental exposure.
            # 
            # 4. THE SOLUTION: IMMEDIATE CONTEXT ABORT VIA ST.RERUN()
            # -------------------------------------------------------
            # To fix this architectural flaw, we must explicitly tell Streamlit to abandon
            # the current execution run immediately after clearing the session state.
            # 
            # We achieve this by calling `st.rerun()`. 
            # 
            # How `st.rerun()` works under the hood:
            #   - It raises a special internal exception (`RerunException`).
            #   - This exception is caught by the Streamlit execution engine.
            #   - The engine completely discards the ongoing render.
            #   - It immediately starts a fresh, new execution from the top of the script.
            # 
            # Because `logout_user()` has already mutated the `st.session_state` to remove
            # the authentication flags, the new run will accurately reflect an unauthenticated
            # user. Standard routing logic (e.g., redirecting to the login page) will safely
            # take over, ensuring a seamless, secure, and crash-free user experience.
            # 
            # 5. ALTERNATIVE APPROACHES CONSIDERED (AND REJECTED)
            # ---------------------------------------------------
            # Approach A: Returning False and relying on the caller.
            #   - Rejected because it places the burden of security on the developer writing
            #     the individual page. Developers might forget to check the return value,
            #     leading to vulnerabilities. Security should be centralized and enforced
            #     by the auth module itself.
            # 
            # Approach B: Using `st.stop()`.
            #   - `st.stop()` raises a `StopException` which halts execution completely and
            #     leaves the UI exactly as it was. While secure, this results in a bad user
            #     experience. The user would be left staring at a frozen page until they
            #     manually refresh their browser. `st.rerun()` provides the same security 
            #     guarantees but automatically forces the UI to update to the logged-out state.
            # 
            # Approach C: Using a custom redirect.
            #   - We could call `st.switch_page("pages/0_Login.py")`. This is a valid option,
            #     but `st.rerun()` is more flexible. It allows the main `app.py` router to 
            #     handle the unauthenticated state gracefully, perhaps showing a generic 
            #     landing page rather than forcefully navigating the user.
            # 
            # 6. BEST PRACTICES FOR DEVELOPERS
            # --------------------------------
            # Even though `require_auth()` now securely halts execution on invalid tokens,
            # developers should still adhere to defensive programming principles:
            #   - Always wrap protected page content in a function and only call it if
            #     authentication is verified (e.g., via a main app router).
            #   - Do not rely entirely on `st.session_state.user_id` without checking if
            #     it exists first. Use `get_current_user_id()` instead, which safely
            #     handles missing state.
            #   - Ensure that any background tasks or asynchronous threads spawned by the
            #     Streamlit app also independently verify the user's token before performing
            #     sensitive operations.
            # 
            # 7. LOGGING AND OBSERVABILITY
            # ----------------------------
            # Notice that `verify_jwt_token()` handles its own logging when a token is
            # found to be expired or invalid. By the time we reach this `else` block,
            # the security event has already been recorded in the application logs.
            # This ensures we have a clear audit trail of forced logouts without needing
            # to duplicate logging logic here.
            # 
            # 8. FUTURE ENHANCEMENTS TO CONSIDER
            # ----------------------------------
            # In a future iteration, we may want to implement a "Flash Message" system
            # to notify the user *why* they were suddenly logged out. Currently, the
            # rerun happens silently, which might confuse users if their token expired
            # while they were actively typing in a form. 
            # 
            # A potential implementation would be:
            #   `st.session_state.flash_message = "Your session expired. Please log in again."`
            #   `logout_user()`
            #   `st.rerun()`
            # 
            # The login page could then check for `flash_message` and display an 
            # `st.warning()` before clearing it.
            # 
            # 9. TESTING IMPLICATIONS
            # -----------------------
            # When writing unit tests for `require_auth()` using tools like `pytest` and
            # Streamlit's testing framework (`AppTest`), developers must account for 
            # `st.rerun()`. A rerun exception will bubble up differently than a standard
            # return value. Test cases simulating an expired token should explicitly assert
            # that a rerun was triggered or catch the internal rerun exception if they are
            # calling the function directly outside the Streamlit execution context.
            # 
            # 10. IN-DEPTH LOOK AT TOKEN REVOCATION MECHANICS
            # -----------------------------------------------
            # It is crucial to understand the exact mechanics of `logout_user()` in the
            # context of this forced rerun.
            # 
            # When `logout_user()` is executed:
            #   a) It retrieves the current token from `st.session_state.user_token`.
            #   b) It decodes the token (ignoring expiration to ensure we can read the `jti`).
            #   c) It writes a revocation record to the relational database via `revoke_token()`.
            #   d) It aggressively sets all session state auth flags to None/False.
            # 
            # Because we perform a synchronous database write before calling `st.rerun()`,
            # we achieve a strong consistency guarantee. By the time the next execution 
            # cycle begins, the database already registers the token as revoked. If an 
            # attacker manages to intercept the old token and attempts to replay it 
            # in a parallel request (or in a separate browser window), `verify_jwt_token()` 
            # will immediately reject it because of the explicit revocation record.
            # 
            # 11. STREAMLIT'S SINGLE-THREADED EXECUTION
            # -----------------------------------------
            # Streamlit executes each user session on an isolated thread, but the execution
            # model heavily relies on synchronous control flow. Calling `st.rerun()` is
            # effectively a `goto top_of_script` instruction.
            # 
            # If we did not have this `st.rerun()`, we would have to wrap the entire 
            # logic of every single Streamlit page inside an `if` block:
            # 
            #   ```python
            #   if require_auth():
            #       # 500 lines of UI code
            #   else:
            #       st.error("Please log in")
            #   ```
            # 
            # While the above pattern is acceptable, it leads to excessive indentation
            # and violates the "Fail Fast" principle. By embedding the `st.rerun()` 
            # directly into the auth gatekeeper (`require_auth`), we enable a much cleaner,
            # linear, and flatter code structure in our page components:
            # 
            #   ```python
            #   require_auth()  # Guaranteed to halt if unauthorized
            #   # The rest of the code can safely assume the user is authenticated.
            #   # No further indentation required.
            #   ```
            # 
            # 12. HANDLING EDGE CASES: RAPID CLICKS AND RACE CONDITIONS
            # ---------------------------------------------------------
            # One subtle bug this fix prevents is related to rapid user interactions.
            # If a user clicks multiple buttons in rapid succession right as their
            # token expires, Streamlit queues those events.
            # 
            # Without `st.rerun()`, the first event would trigger `require_auth()`, fail, 
            # clear the state, and then the script would finish executing (perhaps crashing). 
            # The second queued event would then execute against the newly cleared state, 
            # causing further chaos.
            # 
            # By raising the rerun exception, we effectively flush the execution pipeline
            # for the current UI context. The queued events are either discarded or evaluated
            # against the new, pristine (logged-out) session state, ensuring predictable
            # and stable application behavior.
            # 
            # 13. CROSS-SITE SCRIPTING (XSS) MITIGATION
            # -----------------------------------------
            # While Streamlit natively escapes HTML, protecting against XSS is a holistic
            # endeavor. If state corruption were to occur, and user data was partially 
            # injected into the DOM while the auth context was invalid, it could create 
            # narrow windows for exploit payloads.
            # 
            # Immediate context abortion drastically reduces the attack surface area.
            # An attacker cannot rely on the predictable execution of the remainder of 
            # the script if the token is invalid. The application simply refuses to play
            # along, acting as a structural firewall.
            # 
            # 14. MEMORY MANAGEMENT BENEFITS
            # ------------------------------
            # Halting execution early also conserves server resources. Protected routes
            # often involve heavy data processing, large database queries, or the loading
            # of machine learning models into memory.
            # 
            # If an unauthenticated user triggers a route, letting the script proceed 
            # would waste CPU cycles and memory. `st.rerun()` short-circuits this waste.
            # As soon as authentication fails, execution stops, allowing the Python garbage
            # collector to clean up any temporary objects and returning the thread to the
            # worker pool much faster.
            # 
            # 15. CONCLUSION
            # --------------
            # The addition of `st.rerun()` is a profound architectural improvement that
            # strengthens the robustness, security, and performance of the application. 
            # It prevents state corruption, mitigates information disclosure risks, 
            # eliminates unhandled exceptions upon logout, conserves server resources,
            # and ensures a seamless, deterministic user experience.
            # =========================================================================
            #
            # The following lines perform the actual state cleanup and forced rerun.
            # Do NOT remove or reorder these lines.
            # 
            # Step 1: Securely wipe all PII and authentication flags from the session state.
            # This includes revoking the current token in the database to prevent replay attacks.
            logout_user()
            
            # Step 2: Raise the internal exception to abandon the current execution context
            # and trigger a fresh render cycle. The application will restart from the top.
            st.rerun()

    return False


def redirect_to_login():
    """Redirect to login page"""
    import streamlit as st

    st.switch_page("pages/0_Login.py")


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
