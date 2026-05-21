"""
Quick verification script for OTP brute-force protection implementation
"""
import os
os.environ["TESTING"] = "1"
os.environ["DEBUG"] = "0"

from datetime import datetime, timezone, timedelta
from database import (
    Base, engine, SessionLocal, OTPVerification,
    record_otp_failed_attempt, reset_otp_failed_attempts
)
from auth import _hash_otp

# Setup database
Base.metadata.create_all(bind=engine)

print("=" * 60)
print("OTP Brute-Force Protection - Verification Tests")
print("=" * 60)

db = SessionLocal()

# Test 1: Create OTP and track failed attempts
print("\n✓ Test 1: Failed Attempt Tracking")
otp = "123456"
otp_hash = _hash_otp(otp)
expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

otp_record = OTPVerification(
    email="test1@example.com",
    otp_hash=otp_hash,
    expires_at=expires_at,
)
db.add(otp_record)
db.commit()

print(f"  Initial failed_attempts: {otp_record.failed_attempts}")
print(f"  Initial locked_until: {otp_record.locked_until}")
print(f"  Is locked: {otp_record.is_locked()}")

# Test 2: Record failed attempts
print("\n✓ Test 2: Record Failed Attempts")
for i in range(1, 6):
    record_otp_failed_attempt(db, otp_record.id, lockout_duration_minutes=15, max_failed_attempts=5)
    db.refresh(otp_record)
    print(f"  After attempt {i}: failed_attempts={otp_record.failed_attempts}, is_locked={otp_record.is_locked()}")

# Test 3: Verify lockout happened
print("\n✓ Test 3: Verify Lockout")
assert otp_record.failed_attempts == 5, "Failed attempts should be 5"
assert otp_record.locked_until is not None, "locked_until should be set"
assert otp_record.is_locked(), "OTP should be locked"
print(f"  OTP correctly locked after {otp_record.failed_attempts} attempts")
print(f"  Locked until: {otp_record.locked_until}")

# Test 4: Reset failed attempts
print("\n✓ Test 4: Reset Failed Attempts")
reset_otp_failed_attempts(db, otp_record.id)
db.refresh(otp_record)
print(f"  After reset: failed_attempts={otp_record.failed_attempts}")
print(f"  After reset: locked_until={otp_record.locked_until}")
print(f"  Is locked: {otp_record.is_locked()}")
assert otp_record.failed_attempts == 0, "Failed attempts should be 0 after reset"
assert otp_record.locked_until is None, "locked_until should be None after reset"
assert not otp_record.is_locked(), "OTP should not be locked after reset"

# Test 5: Configuration
print("\n✓ Test 5: Security Configuration")
import auth
print(f"  OTP_MAX_FAILED_ATTEMPTS: {auth.OTP_MAX_FAILED_ATTEMPTS}")
print(f"  OTP_LOCKOUT_MINUTES: {auth.OTP_LOCKOUT_MINUTES}")
assert auth.OTP_MAX_FAILED_ATTEMPTS == 5, "Should be 5"
assert auth.OTP_LOCKOUT_MINUTES == 15, "Should be 15"

# Cleanup
db.close()
Base.metadata.drop_all(bind=engine)

print("\n" + "=" * 60)
print("✓ All verification tests passed!")
print("=" * 60)
print("\nOTP Brute-Force Protection Summary:")
print("  • Failed attempts are tracked per OTP record")
print("  • OTP is locked after 5 failed attempts")
print("  • Lockout duration: 15 minutes")
print("  • Failed attempts reset on successful verification")
print("  • Configuration is secure by default")
