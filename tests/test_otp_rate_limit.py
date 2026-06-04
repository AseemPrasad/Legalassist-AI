"""Regression tests for atomic OTP rate limiting."""

from datetime import datetime, timezone, timedelta

import pytest

import database
from database import Base, OTPVerification, create_otp_verification
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = testing_session()
    try:
        yield db
    finally:
        db.close()


def test_create_otp_verification_uses_atomic_counter(monkeypatch, test_db):
    state = {"count": 0, "keys": []}

    def fake_script(*, keys, args):
        state["count"] += 1
        state["keys"].append(keys[0])
        assert args == [3600]
        return state["count"]

    monkeypatch.setattr("database._get_otp_rate_limit_script", lambda: fake_script)

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    first = create_otp_verification(test_db, "User@Example.com", "hash-1", expires_at, max_requests_per_hour=2)
    second = create_otp_verification(test_db, "User@Example.com", "hash-2", expires_at, max_requests_per_hour=2)

    assert first.id != second.id
    assert test_db.query(OTPVerification).count() == 2
    assert "@" not in state["keys"][0]
    assert state["keys"][0] == state["keys"][1]

    with pytest.raises(ValueError, match="Too many OTP requests"):
        create_otp_verification(test_db, "User@Example.com", "hash-3", expires_at, max_requests_per_hour=2)

    assert test_db.query(OTPVerification).count() == 2
