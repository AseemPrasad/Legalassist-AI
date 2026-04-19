"""
Authentication system for LegalAssist AI.
Email-based OTP authentication with JWT session management.
"""

import os
import hashlib
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import logging

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
    User,
)

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 7 * 24  # 7 days

OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "10"))
OTP_RATE_LIMIT_HOURS = 1
OTP_RATE_LIMIT_MAX = 3  # Max OTP requests per email per hour


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
            logger.warning("SendGrid API key not configured, logging OTP instead")
            logger.info(f"OTP for {email}: {otp}")
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
        # Fallback: log OTP for development
        logger.info(f"OTP for {email}: {otp}")
        return False


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
        
        # DUMMY CREDENTIALS FOR TESTING
        if email.lower() == "test@example.com":
            otp = "123456"
            otp_hash = _hash_otp(otp)
            expires_at = now + timedelta(minutes=OTP_EXPIRY_MINUTES)
            create_otp_verification(db, email, otp_hash, expires_at)
            user = get_user_by_email(db, email)
            if not user:
                create_user(db, email)
            return True, "Dummy OTP sent"

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
    Verify OTP and create JWT token.
    Returns (success, message, token).
    """
    db = SessionLocal()
    try:
        # Get pending OTP
        otp_record = get_pending_otp(db, email)

        if not otp_record:
            return False, "OTP expired or not found. Please request a new one.", None

        # Verify OTP
        if not _verify_otp_hash(otp, otp_record.otp_hash):
            return False, "Invalid OTP code. Please try again.", None

        # Mark OTP as used
        mark_otp_as_used(db, otp_record.id)

        # Get or create user
        user = get_user_by_email(db, email)
        if not user:
            user = create_user(db, email)

        # Update last login
        update_user_last_login(db, user.id)

        # Create JWT token
        token = create_jwt_token(user.id, user.email)

        logger.info(f"User logged in: {email} (user_id={user.id})")
        return True, "Login successful", token

    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {str(e)}")
        return False, f"Error: {str(e)}", None
    finally:
        db.close()


def create_jwt_token(user_id: int, email: str) -> str:
    """Create JWT token for authenticated user"""
    payload = {
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
    Returns None if token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        return None


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
    Cleanup expired OTPs and old data.
    Returns count of cleaned up records.
    """
    db = SessionLocal()
    try:
        deleted = cleanup_expired_otps(db)
        logger.info(f"Cleaned up {deleted} expired OTPs")
        return deleted
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
            st.session_state.user_id = payload.get("user_id")

        st.session_state.is_authenticated = True
        st.session_state.pending_email = None
        st.session_state.otp_sent = False

        return True

    return False


def logout_user():
    """Logout current user"""
    import streamlit as st

    init_auth_session()
    st.session_state.user_token = None
    st.session_state.user_email = None
    st.session_state.user_id = None
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
            # Token expired, logout
            logout_user()

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
