import types
from types import SimpleNamespace

import pytest

from db.session import init_db, get_db_with_rls, db_session
from db.base import Base
from db.models.auth import User
from db.models.cases import Case


def make_request_with_user(user_id: int):
    req = SimpleNamespace()
    req.state = SimpleNamespace()
    req.state.db_rls_user_id = user_id
    return req


def test_rls_prevents_cross_tenant_case_access(tmp_path, monkeypatch):
    # Use an in-memory SQLite for isolation from the developer DB schema
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    # Patch db.session to use the in-memory engine/session for this test
    monkeypatch.setattr('db.session.engine', engine)
    monkeypatch.setattr('db.session.SessionLocal', SessionLocal)

    # Create the schema in-memory
    from db.base import Base
    Base.metadata.create_all(bind=engine)

    # Create two cases belonging to different user_ids.
    with db_session() as db:
        case1 = Case(user_id=1001, case_number="A-1", case_type="civil", jurisdiction="X", title="Case 1")
        case2 = Case(user_id=2002, case_number="B-1", case_type="civil", jurisdiction="Y", title="Case 2")
        db.add(case1)
        db.add(case2)
        db.commit()
        db.refresh(case1)
        db.refresh(case2)
        c1_id = case1.id
        c2_id = case2.id

    # Simulate request context for user1
    req = make_request_with_user(1001)
    db = get_db_with_rls(req)

    # user1 should see their own case
    found = db.query(Case).filter(Case.id == c1_id).first()
    assert found is not None and found.user_id == 1001

    # user1 should NOT see user2's case thanks to RLS/app-scoping
    found2 = db.query(Case).filter(Case.id == c2_id).first()
    assert found2 is None

    db.close()