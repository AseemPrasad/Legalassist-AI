import os
from datetime import datetime, timezone
from config import Config
import services.case_anonymization as ca

import pytest


def test_anonymized_id_changes_with_secret(monkeypatch):
    # Import inside test so monkeypatch can affect env var used during module import.
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "a" * 40)

    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    anon_a = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "b" * 40)
    anon_b = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    assert anon_a != anon_b


def test_anonymized_id_deterministic_with_same_secret(monkeypatch):
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "s" * 40)

    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    anon_1 = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)
    anon_2 = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    assert anon_1 == anon_2


def test_anonymized_id_format(monkeypatch):
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "f" * 40)

    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    anon = ca._generate_anonymized_case_id(case_id=123, created_at=created_at)

    assert len(anon) == 12
    int(anon, 16)  # should be hex


def test_different_inputs_produce_different_ids(monkeypatch):
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "z" * 40)
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
        # should not raise when testing
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

