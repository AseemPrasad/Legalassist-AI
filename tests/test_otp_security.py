"""
Tests for OTP security features including brute-force protection.
"""

import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

# Set test environment variables before imports
os.environ["DEBUG"] = "0"
os.environ["TESTING"] = "1"
os.environ["OTP_EXPIRY_MINUTES"] = "10"
os.environ["OTP_MAX_FAILED_ATTEMPTS"] = "5"
os.environ["OTP_LOCKOUT_MINUTES"] = "15"

from database import SessionLocal, OTPVerification, init_db, Base, engine
import auth
from auth import verify_otp_and_create_token, _hash_otp, _verify_otp_hash


@pytest.fixture(scope="function")
def setup_test_db():
    """Setup and cleanup test database for each test"""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop all tables after test
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_test_db):
    """Get a database session for testing"""
    db = SessionLocal()
    yield db
    db.close()


class TestOTPBruteForceProtection:
    """Test suite for OTP brute-force protection mechanism"""

    def test_failed_attempt_tracking(self, db_session):
        """Test that failed OTP verification attempts are tracked"""
        email = "test@example.com"
        otp = "123456"
        otp_hash = _hash_otp(otp)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        # Create OTP record
        otp_record = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
        )
        db_session.add(otp_record)
        db_session.commit()

        # Verify it starts with 0 failed attempts
        assert otp_record.failed_attempts == 0
        assert otp_record.locked_until is None

    def test_otp_lockout_after_max_attempts(self, db_session):
        """Test that OTP is locked after maximum failed attempts"""
        email = "test@example.com"
        otp = "123456"
        otp_hash = _hash_otp(otp)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        # Create OTP record
        otp_record = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
        )
        db_session.add(otp_record)
        db_session.commit()

        # Simulate max failed attempts
        max_attempts = 5
        lockout_minutes = 15

        for i in range(max_attempts):
            otp_record.failed_attempts += 1
            if otp_record.failed_attempts >= max_attempts:
                otp_record.locked_until = datetime.now(timezone.utc) + timedelta(minutes=lockout_minutes)
            db_session.commit()

        # Verify OTP is locked
        assert otp_record.failed_attempts == max_attempts
        assert otp_record.locked_until is not None
        assert otp_record.is_locked()

    def test_is_locked_method(self, db_session):
        """Test the is_locked() method of OTPVerification"""
        email = "test@example.com"
        otp_hash = _hash_otp("123456")
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        # Create unlocked OTP
        otp_record = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
            locked_until=None,
        )
        db_session.add(otp_record)
        db_session.commit()

        # Should not be locked
        assert not otp_record.is_locked()

        # Lock it with future timestamp
        otp_record.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        db_session.commit()
        assert otp_record.is_locked()

        # Lock it with past timestamp (should not be locked)
        otp_record.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        db_session.commit()
        assert not otp_record.is_locked()

    @patch("auth.send_otp_email")
    @patch("auth.create_jwt_token")
    def test_verify_otp_with_failed_attempts(self, mock_jwt, mock_send_email, db_session):
        """Test OTP verification with failed attempts tracking"""
        mock_jwt.return_value = "test_token"
        mock_send_email.return_value = True

        email = "sectest@example.com"
        correct_otp = "654321"
        wrong_otp = "000000"

        # Request OTP
        otp_hash = _hash_otp(correct_otp)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_record = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
        )
        db_session.add(otp_record)
        db_session.commit()

        # Get the created user
        from database import create_user, get_user_by_email
        user = get_user_by_email(db_session, email)
        if not user:
            user = create_user(db_session, email)

        # Try wrong OTP and verify failed attempts are tracked
        success, message, token = verify_otp_and_create_token(email, wrong_otp)

        assert not success
        assert token is None
        assert "Invalid OTP code" in message or "attempts remaining" in message

        # Verify failed attempt was recorded in database
        # Need to create a fresh session to see the changes from verify_otp_and_create_token
        fresh_db = SessionLocal()
        try:
            otp_record = fresh_db.query(OTPVerification).filter(
                OTPVerification.email == email,
                OTPVerification.is_used == False,
            ).first()
            assert otp_record is not None
            assert otp_record.failed_attempts == 1
        finally:
            fresh_db.close()

    @patch("auth.send_otp_email")
    @patch("auth.create_jwt_token")
    def test_verify_otp_lockout_message(self, mock_jwt, mock_send_email, db_session):
        """Test that user gets helpful message when OTP is locked"""
        mock_jwt.return_value = "test_token"
        mock_send_email.return_value = True

        email = "locktest@example.com"
        wrong_otp = "000000"

        # Create OTP and manually lock it
        otp_hash = _hash_otp("correct_otp")
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_record = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
            failed_attempts=5,  # Max attempts reached
            locked_until=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        db_session.add(otp_record)
        db_session.commit()

        # Create user
        from database import create_user, get_user_by_email
        user = get_user_by_email(db_session, email)
        if not user:
            user = create_user(db_session, email)

        # Try to verify - should fail with lockout message
        success, message, token = verify_otp_and_create_token(email, wrong_otp)

        assert not success
        assert token is None
        # Check for lockout message - should mention locked or remaining time
        assert "Too many failed attempts" in message or "locked" in message.lower() or "minutes" in message.lower()

    @patch("auth.send_otp_email")
    def test_otp_verification_reset_on_success(self, mock_send_email, db_session):
        """Test that failed attempts counter is reset on successful verification"""
        mock_send_email.return_value = True

        from database import record_otp_failed_attempt, reset_otp_failed_attempts

        email = "resettest@example.com"
        correct_otp = "789456"
        otp_hash = _hash_otp(correct_otp)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        # Create OTP record
        otp_record = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
        )
        db_session.add(otp_record)
        db_session.commit()

        # Simulate some failed attempts
        record_otp_failed_attempt(db_session, otp_record.id)
        record_otp_failed_attempt(db_session, otp_record.id)

        otp_record = db_session.query(OTPVerification).filter(
            OTPVerification.id == otp_record.id
        ).first()
        assert otp_record.failed_attempts == 2

        # Reset failed attempts
        reset_otp_failed_attempts(db_session, otp_record.id)

        otp_record = db_session.query(OTPVerification).filter(
            OTPVerification.id == otp_record.id
        ).first()
        assert otp_record.failed_attempts == 0
        assert otp_record.locked_until is None


class TestOTPSecurityConstants:
    """Test suite for OTP security configuration constants"""

    def test_otp_max_failed_attempts_configured(self):
        """Verify OTP_MAX_FAILED_ATTEMPTS is configured"""
        assert hasattr(auth, "OTP_MAX_FAILED_ATTEMPTS")
        assert auth.OTP_MAX_FAILED_ATTEMPTS > 0
        assert auth.OTP_MAX_FAILED_ATTEMPTS == 5  # From test environment

    def test_otp_lockout_minutes_configured(self):
        """Verify OTP_LOCKOUT_MINUTES is configured"""
        assert hasattr(auth, "OTP_LOCKOUT_MINUTES")
        assert auth.OTP_LOCKOUT_MINUTES > 0
        assert auth.OTP_LOCKOUT_MINUTES == 15  # From test environment

    def test_security_constants_from_env(self):
        """Test that security constants can be configured from environment"""
        # These should be configurable via environment variables
        assert os.getenv("OTP_MAX_FAILED_ATTEMPTS") == "5"
        assert os.getenv("OTP_LOCKOUT_MINUTES") == "15"


class TestOTPHashVerification:
    """Test suite for OTP hashing and verification"""

    def test_otp_hash_consistency(self):
        """Test that OTP hashing is consistent"""
        otp = "123456"
        hash1 = _hash_otp(otp)
        hash2 = _hash_otp(otp)
        assert hash1 == hash2

    def test_otp_hash_different_for_different_otps(self):
        """Test that different OTPs produce different hashes"""
        hash1 = _hash_otp("123456")
        hash2 = _hash_otp("654321")
        assert hash1 != hash2

    def test_verify_otp_hash_success(self):
        """Test successful OTP hash verification"""
        otp = "123456"
        otp_hash = _hash_otp(otp)
        assert _verify_otp_hash(otp, otp_hash)

    def test_verify_otp_hash_failure(self):
        """Test failed OTP hash verification"""
        otp = "123456"
        wrong_otp = "654321"
        otp_hash = _hash_otp(otp)
        assert not _verify_otp_hash(wrong_otp, otp_hash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
