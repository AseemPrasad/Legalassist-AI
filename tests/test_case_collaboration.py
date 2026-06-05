import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database
import case_manager
from database import Base, User, Case, CaseStatus
from db.crud.comments import get_case_comments


@pytest.fixture()
def collaboration_db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

    monkeypatch.setattr(database, "SessionLocal", TestSession)
    monkeypatch.setattr(case_manager, "SessionLocal", TestSession)

    session = TestSession()
    user = User(email="collab@example.com")
    session.add(user)
    session.commit()
    session.refresh(user)

    case = Case(
        user_id=user.id,
        case_number="CASE-100",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Collaboration Test Case",
    )
    session.add(case)
    session.commit()
    session.refresh(case)

    user_id = user.id
    case_id = case.id
    session.close()
    return TestSession, user_id, case_id


def test_case_collaboration_comment_and_presence(collaboration_db):
    TestSession, user_id, case_id = collaboration_db

    comment = case_manager.add_case_comment(
        user_id=user_id,
        case_id=case_id,
        comment_text="Please review the deadline and supporting affidavit.",
        active_view="collaboration",
    )

    assert comment is not None
    assert comment.comment_text.startswith("Please review")

    detail = case_manager.get_case_detail(user_id, case_id)
    assert detail is not None
    assert len(detail["comments"]) == 1
    assert detail["comments"][0]["comment_text"].startswith("Please review")
    assert len(detail["presence"]) == 1
    assert detail["presence"][0]["user_id"] == user_id
    assert any(item["event_type"] in {"comment_added", "comment_replied"} for item in detail["timeline"])


def test_get_case_comments_denies_unauthorized_user(collaboration_db):
    """get_case_comments must raise PermissionError when called by a user
    who does not own the requested case, regardless of whether the caller
    performs an external ownership check.  This verifies that access-control
    is enforced at the retrieval layer itself.
    """
    TestSession, owner_user_id, case_id = collaboration_db

    # Create a second user who does NOT own the case.
    session = TestSession()
    other_user = User(email="intruder@example.com")
    session.add(other_user)
    session.commit()
    session.refresh(other_user)
    intruder_id = other_user.id
    session.close()

    # Directly calling get_case_comments with the wrong user_id must be denied
    # even without any outer ownership guard.
    with TestSession() as db:
        with pytest.raises(PermissionError):
            get_case_comments(db, case_id, intruder_id)
