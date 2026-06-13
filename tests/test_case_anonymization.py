import os
from datetime import datetime, timezone
from config import Config
import services.case_anonymization as ca

import pytest

# Use secrets with sufficient entropy (8+ distinct chars) to pass the new
# entropy check introduced by the #1240 fix.
_SECRET_A = "aB3dEf7hIj0kLmNoPqRsTuVwXyZ12345678"
_SECRET_B = "bC4eFg8iJk1lMnOpQrStUvWxYz23456789a"
_SECRET_S = "sT5uGh9jKl2mNoOpQrStUvWxYz34567890b"
_SECRET_F = "fG6vHi0kLm3nOpQrStUvWxYz45678901cd"
_SECRET_Z = "zA7wBj1lMn4oOpQrStUvWxYz56789012ef"


def test_anonymized_id_changes_with_secret(monkeypatch):
    # Import inside test so monkeypatch can affect env var used during module import.
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _SECRET_A)

    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    anon_a = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _SECRET_B)
    anon_b = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    assert anon_a != anon_b


def test_anonymized_id_deterministic_with_same_secret(monkeypatch):
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _SECRET_S)

    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    anon_1 = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)
    anon_2 = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    assert anon_1 == anon_2


def test_anonymized_id_format(monkeypatch):
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _SECRET_F)

    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    anon = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    assert len(anon) == 12
    int(anon, 16)  # should be hex


def test_different_inputs_produce_different_ids(monkeypatch):
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _SECRET_Z)
    from datetime import datetime, timezone

    dt1 = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    dt2 = datetime(2024, 1, 3, 3, 4, 5, tzinfo=timezone.utc)

    id1 = ca._generate_anonymized_case_id(case_id=10, created_at=dt1)
    id2 = ca._generate_anonymized_case_id(case_id=11, created_at=dt1)
    id3 = ca._generate_anonymized_case_id(case_id=10, created_at=dt2)

    assert id1 != id2
    assert id1 != id3


def test_secret_override_allowed_only_in_testing():
    # Temporarily toggle testing flag
    prev = Config.TESTING
    try:
        Config.TESTING = True
        # should not raise when testing — override bypasses entropy check
        dt = __import__("datetime").datetime.utcnow()
        val = ca._generate_anonymized_case_id(1, dt, secret_override=("o" * 40))
        assert isinstance(val, str) and len(val) == 12
    finally:
        Config.TESTING = prev

    # when not testing, override should raise
    prev = Config.TESTING
    try:
        Config.TESTING = False
        with pytest.raises(RuntimeError):
            ca._generate_anonymized_case_id(1, dt, secret_override=("o" * 40))
    finally:
        Config.TESTING = prev
