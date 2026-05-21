import os
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

import jwt
import pytest

os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("JWT_SECRET", "test_secret_key")
os.environ.setdefault("JWT_ISSUER", "legalassist.ai")
os.environ.setdefault("JWT_AUDIENCE", "legalassist-users")

import auth
import database
from database import Base, OTPVerification
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class DummyUser(SimpleNamespace):
    pass


@pytest.fixture()
def test_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(auth, "SessionLocal", testing_session)
    monkeypatch.setattr(database, "SessionLocal", testing_session)
    db = testing_session()
    try:
        yield db
    finally:
        db.close()


def _valid_payload():
    return {
        "sub": "123",
        "user_id": 123,
        "email": "tester@example.com",
        "role": "user",
    }


def test_verify_jwt_token_rejects_bad_claims(monkeypatch, test_db):
    monkeypatch.setattr(auth, "is_token_revoked", lambda db, jti: False)
    monkeypatch.setattr(auth, "get_user_by_email", lambda db, email: DummyUser(id=1, email=email))

    now = datetime.now(timezone.utc)

    token_missing_type = jwt.encode(
        {
            **_valid_payload(),
            "jti": "jti-1",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "iss": auth.Config.JWT_ISSUER,
            "aud": auth.Config.JWT_AUDIENCE,
        },
        auth.JWT_SECRET,
        algorithm=auth.JWT_ALGORITHM,
    )
    assert auth.verify_jwt_token(token_missing_type) is None

    token_bad_issuer = jwt.encode(
        {
            **_valid_payload(),
            "jti": "jti-2",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "iss": "other-issuer",
            "aud": auth.Config.JWT_AUDIENCE,
            "type": "access",
        },
        auth.JWT_SECRET,
        algorithm=auth.JWT_ALGORITHM,
    )
    assert auth.verify_jwt_token(token_bad_issuer) is None

    token_bad_audience = jwt.encode(
        {
            **_valid_payload(),
            "jti": "jti-3",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "iss": auth.Config.JWT_ISSUER,
            "aud": "wrong-audience",
            "type": "access",
        },
        auth.JWT_SECRET,
        algorithm=auth.JWT_ALGORITHM,
    )
    assert auth.verify_jwt_token(token_bad_audience) is None


def test_verify_jwt_token_rejects_revoked_token(monkeypatch, test_db):
    token = auth.create_jwt_token(123, "tester@example.com")

    monkeypatch.setattr(auth, "get_user_by_email", lambda db, email: DummyUser(id=1, email=email))

    revoked_jti = jwt.decode(
        token,
        auth.JWT_SECRET,
        algorithms=[auth.JWT_ALGORITHM],
        audience=auth.Config.JWT_AUDIENCE,
        issuer=auth.Config.JWT_ISSUER,
        options={"verify_exp": False},
    )["jti"]

    monkeypatch.setattr(auth, "is_token_revoked", lambda db, jti: jti == revoked_jti)

    assert auth.verify_jwt_token(token) is None


def test_verify_otp_rejects_reuse(monkeypatch, test_db):
    monkeypatch.setattr(auth, "SessionLocal", lambda: test_db)
    monkeypatch.setattr(database, "SessionLocal", lambda: test_db)
    monkeypatch.setattr(auth, "create_jwt_token", lambda user_id, email: "token")

    email = "reuse@example.com"
    otp = "123456"
    otp_hash = auth._hash_otp(otp)
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    db_otp = OTPVerification(email=email, otp_hash=otp_hash, expires_at=expires)
    test_db.add(db_otp)
    test_db.commit()

    first_success, _, token = auth.verify_otp_and_create_token(email, otp)
    assert first_success is True
    assert token == "token"

    second_success, message, second_token = auth.verify_otp_and_create_token(email, otp)
    assert second_success is False
    assert second_token is None
    assert "request a new one" in message.lower() or "not found" in message.lower()


def test_create_otp_verification_rate_limits_email_and_ip(monkeypatch, test_db):
    state = {"calls": []}

    def fake_script(*, keys, args):
        state["calls"].append(keys[0])
        return len(state["calls"])

    monkeypatch.setattr("db.otp_service._get_otp_rate_limit_script", lambda: fake_script)

    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    database.create_otp_verification(
        test_db,
        "User@Example.com",
        "hash-1",
        expires,
        max_requests_per_hour=5,
        requester_ip="203.0.113.10",
    )

    assert len(state["calls"]) == 2
    assert any(key.startswith("otp:rate:") for key in state["calls"])
