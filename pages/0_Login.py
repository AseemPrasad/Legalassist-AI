"""
Login/Signup Page for LegalAssist AI.
Email-based OTP authentication.
"""

import streamlit as st
from datetime import datetime, timezone
import time

from auth import (
    init_auth_session,
    login_user,
    verify_login,
    logout_user,
    require_auth,
    get_current_user_email,
    verify_jwt_token,
)

# Page config
st.set_page_config(
    page_title="Login - LegalAssist AI",
    page_icon="🔐",
    layout="centered",
)

# No custom styling, using Streamlit default theme


def render_login_card():
    """Render the login card UI"""
    st.title("⚖️ LegalEase AI")
    st.subheader("Secure Case History & Timeline Tracking")

    with st.container(border=True):
        st.markdown("### 🔐 Login / Register")
        # Email input
        email = st.text_input(
            "Email Address",
            placeholder="your@email.com",
            key="login_email",
            help="We'll send a 6-digit OTP to this email",
        )

        st.caption("💡 Hint: Use **test@example.com** for a quick dummy account login")

        if st.button("📧 Send OTP", use_container_width=True):
            if not email:
                st.error("Please enter your email address")
            else:
                with st.spinner("Sending OTP to your email..."):
                    success = login_user(email)
                    if success:
                        st.success("✅ OTP sent! Check your email (and spam folder).")
                        st.rerun()
                    else:
                        st.error("Failed to send OTP. Please try again.")

        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #888; font-size: 0.9rem;">
                <p>🔒 Password-free authentication</p>
                <p>We'll send a one-time code to your email</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_otp_verification():
    """Render OTP verification UI"""
    email = st.session_state.get("pending_email", "")
    st.title("⚖️ Verify OTP")
    st.subheader(f"Enter the 6-digit code sent to {email}")

    with st.container(border=True):
        # OTP input
        otp = st.text_input(
            "OTP Code",
            placeholder="123456",
            key="login_otp",
            max_chars=6,
            help="Enter the 6-digit code from your email",
        )

        if email.lower() == "test@example.com":
            st.caption("💡 Hint: Dummy OTP is **123456**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔓 Verify", use_container_width=True):
                if not otp or len(otp) != 6:
                    st.error("Please enter a valid 6-digit OTP")
                else:
                    with st.spinner("Verifying..."):
                        success = verify_login(otp)
                        if success:
                            st.success("✅ Login successful! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid OTP. Please try again.")

        with col2:
            if st.button("🔄 Resend", use_container_width=True):
                with st.spinner("Sending new OTP..."):
                    success = login_user(email)
                    if success:
                        st.success("New OTP sent!")
                        st.rerun()
                    else:
                        st.error("Failed to resend OTP.")

        st.markdown("---")

        if st.button("← Use different email", use_container_width=True):
            st.session_state.pending_email = None
            st.session_state.otp_sent = False
            st.rerun()


def render_logged_in_state():
    """Render UI for already logged-in user"""
    email = get_current_user_email()

    st.title("⚖️ Welcome Back!")
    st.subheader(f"Logged in as {email}")

    with st.container(border=True):
        st.success("✅ You are actively logged in")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Go to Dashboard", use_container_width=True):
                st.switch_page("pages/1_My_Cases.py")

        with col2:
            if st.button("🚀 Upload Judgment", use_container_width=True):
                st.switch_page("app.py")

        st.markdown("---")

        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            logout_user()
            st.rerun()


def main():
    """Main login page logic"""
    init_auth_session()

    # Check if already logged in
    if require_auth():
        render_logged_in_state()
        return

    # Check if OTP was sent and waiting for verification
    if st.session_state.get("otp_sent"):
        render_otp_verification()
    else:
        render_login_card()

    # Info section
    st.markdown("---")

    st.info(
        """
        ### How Login Works

        1. **Enter your email** - No password needed
        2. **Get OTP via email** - One-time code sent instantly
        3. **Enter OTP** - Secure, password-free login
        4. **Stay logged in** - Session lasts 7 days

        **Benefits of having an account:**
        - Track multiple cases over time
        - See timeline of all your documents
        - Never miss important deadlines
        - Export case summaries as PDF
        - Share anonymized cases with lawyers
        """
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.85rem;">
            <p>By logging in, you agree to our Terms of Service and Privacy Policy.</p>
            <p>Your data is encrypted and stored securely.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
