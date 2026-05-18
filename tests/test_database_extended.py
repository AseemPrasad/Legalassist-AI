
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from database import (
    Base,
    CaseRecord,
    CaseOutcome,
    UserFeedback,
    User,
    OTPVerification,
    Case,
    CaseStatus,
    DocumentType,
    CaseDocument,
    NotificationLog,
    RevokedToken,
    init_db,
    create_case_record,
    update_case_outcome,
    get_case_record,
    get_cases_by_criteria,
    submit_user_feedback,
    get_user_feedback,
    get_user_by_email,
    create_user,
    update_user_last_login,
    create_otp_verification,
    get_pending_otp,
    mark_otp_as_used,
    cleanup_expired_otps,
    revoke_token,
    cleanup_expired_revoked_tokens,
    is_token_revoked,
    create_case,
    get_user_cases,
    get_case_by_id,
    get_case_by_number,
    update_case_status,
    delete_case,
    create_case_document,
    create_timeline_event,
    create_case_deadline,
    get_user_deadlines,
    get_upcoming_deadlines,
)


@pytest.fixture(autouse=True)
def disable_otp_rate_limiter(monkeypatch):
    monkeypatch.setattr("db.otp_service._reserve_otp_rate_limit_slot", lambda *args, **kwargs: True)

@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

class TestDatabaseExtended:
    """Extended tests for database.py to improve coverage"""

    def test_repr_methods(self, test_db):
        """Test __repr__ methods of all models for coverage"""
        # CaseRecord
        record = CaseRecord(hashed_case_id="C1", case_type="civil", jurisdiction="Delhi", outcome="won")
        assert "CaseRecord" in repr(record)
        
        # CaseOutcome
        outcome = CaseOutcome(case_id=1, appeal_filed=True, appeal_success=False)
        assert "CaseOutcome" in repr(outcome)
        
        # UserFeedback
        feedback = UserFeedback(user_id=1, appeal_outcome="lost")
        assert "UserFeedback" in repr(feedback)
        
        # User
        user = User(email="test@example.com")
        assert "User" in repr(user)
        
        # Case
        case = Case(case_number="CN1", status=CaseStatus.ACTIVE)
        assert "Case" in repr(case)
        
        # CaseDocument
        doc = CaseDocument(case_id=1, document_type=DocumentType.JUDGMENT)
        assert "CaseDocument" in repr(doc)

    def test_case_record_operations(self, test_db):
        """Test CaseRecord creation and retrieval"""
        record = create_case_record(
            test_db, "C1", "civil", "Delhi", 
            court_name="High Court", outcome="plaintiff_won"
        )
        assert record.hashed_case_id == "C1"
        
        retrieved = get_case_record(test_db, "C1")
        assert retrieved.id == record.id
        
        # Criteria search
        results = get_cases_by_criteria(test_db, case_type="civil", jurisdiction="Delhi")
        assert len(results) == 1
        
        results = get_cases_by_criteria(test_db, jurisdiction="Mumbai")
        assert len(results) == 0

    def test_deadline_queries_handle_naive_datetimes(self, test_db):
        """Test deadline comparisons and ordering when stored values are naive."""
        now = datetime.now(timezone.utc)

        case1 = Case(
            user_id=1,
            case_number="CASE-NAIVE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case Naive 1",
        )
        case2 = Case(
            user_id=1,
            case_number="CASE-NAIVE-2",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case Naive 2",
        )
        test_db.add_all([case1, case2])
        test_db.commit()
        test_db.refresh(case1)
        test_db.refresh(case2)

        deadline_one = create_case_deadline(
            test_db,
            1,
            case1.id,
            "Case Naive 1",
            now + timedelta(days=8),
            "appeal",
        )
        deadline_two = create_case_deadline(
            test_db,
            1,
            case2.id,
            "Case Naive 2",
            now + timedelta(days=18),
            "filing",
        )

        deadline_one.deadline_date = deadline_one.deadline_date.replace(tzinfo=None)
        deadline_two.deadline_date = deadline_two.deadline_date.replace(tzinfo=None)
        test_db.commit()

        upcoming = get_upcoming_deadlines(test_db, days_before=30)
        assert [deadline.id for deadline in upcoming] == [deadline_one.id, deadline_two.id]

        user_deadlines = get_user_deadlines(test_db, 1)
        assert [deadline.id for deadline in user_deadlines] == [deadline_one.id, deadline_two.id]

    def test_case_outcome_operations(self, test_db):
        """Test CaseOutcome updates"""
        record = create_case_record(test_db, "C1", "civil", "Delhi", outcome="won")
        
        outcome = update_case_outcome(
            test_db, "C1", appeal_filed=True, appeal_success=True, appeal_cost="5000"
        )
        assert outcome.appeal_filed == True
        assert outcome.appeal_success == True
        
        # Test error for non-existent case
        with pytest.raises(ValueError):
            update_case_outcome(test_db, "NON_EXISTENT")

    def test_user_feedback_operations(self, test_db):
        """Test UserFeedback submission and retrieval"""
        feedback = submit_user_feedback(
            test_db, 1, did_appeal=True, appeal_outcome="won", satisfaction_rating=5
        )
        assert feedback.user_id == 1
        
        history = get_user_feedback(test_db, 1)
        assert len(history) == 1
        assert history[0].appeal_outcome == "won"

    def test_user_authentication_operations(self, test_db):
        """Test User and OTP operations"""
        # User
        user = create_user(test_db, "auth@example.com")
        assert user.email == "auth@example.com"
        
        retrieved = get_user_by_email(test_db, "auth@example.com")
        assert retrieved.id == user.id
        
        updated = update_user_last_login(test_db, user.id)
        assert updated.last_login is not None
        
        # OTP
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=10)
        otp = create_otp_verification(test_db, "auth@example.com", "hash", expires)
        
        pending = get_pending_otp(test_db, "auth@example.com")
        assert pending.id == otp.id
        
        mark_otp_as_used(test_db, otp.id)
        assert get_pending_otp(test_db, "auth@example.com") is None
        
        # Cleanup
        create_otp_verification(test_db, "expired@example.com", "h", now - timedelta(minutes=1))
        deleted = cleanup_expired_otps(test_db)
        assert deleted >= 1

    def test_case_management_operations(self, test_db):
        """Test Case and Document operations"""
        user = create_user(test_db, "case@example.com")
        
        case = create_case(test_db, user.id, "CASE-001", "criminal", "Delhi")
        assert case.case_number == "CASE-001"
        
        cases = get_user_cases(test_db, user.id)
        assert len(cases) == 1
        
        by_id = get_case_by_id(test_db, case.id)
        assert by_id.case_number == "CASE-001"
        
        by_num = get_case_by_number(test_db, user.id, "CASE-001")
        assert by_num.id == case.id
        
        update_case_status(test_db, case.id, CaseStatus.CLOSED)
        assert get_case_by_id(test_db, case.id).status == CaseStatus.CLOSED
        
        # Document
        doc = create_case_document(test_db, case.id, DocumentType.JUDGMENT, user.id, "Some content")
        assert doc.document_type == DocumentType.JUDGMENT
        
        # Delete
        success = delete_case(test_db, case.id)
        assert success == True
        assert get_case_by_id(test_db, case.id) is None

    def test_create_timeline_event(self, test_db):
        """Timeline events should be created without leaking document helper state."""
        user = create_user(test_db, "timeline@example.com")
        case = create_case(test_db, user.id, "CASE-TL-001", "civil", "Delhi")

        event = create_timeline_event(
            test_db,
            case.id,
            "status_changed",
            "Case moved to hearing stage",
            metadata={"previous_status": "filed", "new_status": "hearing"},
        )

        assert event.case_id == case.id
        assert event.event_type == "status_changed"
        assert event.description == "Case moved to hearing stage"
        assert event.event_metadata == {"previous_status": "filed", "new_status": "hearing"}
        assert event.id is not None

    def test_init_db(self):
        """Test database initialization (schema creation)"""
        # This just ensures metadata.create_all doesn't crash
        init_db()

    def test_init_db_creates_notification_status_index(self, monkeypatch):
        """init_db should create the NotificationLog.status index for existing databases."""
        from database import Base, NotificationLog, init_db

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        with engine.begin() as connection:
            connection.exec_driver_sql("DROP INDEX IF EXISTS ix_notification_logs_status")

        index_names_before = {index["name"] for index in inspect(engine).get_indexes(NotificationLog.__tablename__)}
        assert "ix_notification_logs_status" not in index_names_before

        monkeypatch.setattr("database.engine", engine)
        init_db()

        index_names_after = {index["name"] for index in inspect(engine).get_indexes(NotificationLog.__tablename__)}
        assert "ix_notification_logs_status" in index_names_after


class TestAutoDeadlineDeduplication:
    """Regression tests for duplicate deadline creation (PR #197 fix)."""

    def test_skip_duplicate_does_not_raise_nameerror(self, test_db):
        """
        Regression: when an existing appeal deadline is found, _auto_create_deadlines_from_remedies
        must NOT try to access `deadline.id` (NameError) — it should log and return cleanly.
        """
        from database import CaseDeadline
        from case_manager import _auto_create_deadlines_from_remedies

        # Pre-insert a deadline that will trigger the "existing deadline" branch
        now = datetime.now(timezone.utc)
        existing = CaseDeadline(
            user_id=1,
            case_id=99,
            case_title="CASE-DUP",
            deadline_date=(now + timedelta(days=30)).replace(tzinfo=None),
            deadline_type="appeal",
            description="Pre-existing appeal deadline",
        )
        test_db.add(existing)
        test_db.commit()
        test_db.refresh(existing)

        remedies = {"appeal_days": "30", "appeal_court": "High Court"}

        # Must not raise NameError or any other exception
        _auto_create_deadlines_from_remedies(
            db=test_db,
            user_id=1,
            case_id=99,
            case_title="CASE-DUP",
            remedies=remedies,
            document_id=42,
        )

        # Only the original deadline should exist — no duplicate created
        deadlines = test_db.query(CaseDeadline).filter(
            CaseDeadline.case_id == 99,
            CaseDeadline.deadline_type == "appeal",
        ).all()
        assert len(deadlines) == 1, "Duplicate deadline must not be created"

    def test_new_deadline_creates_timeline_event(self, test_db):
        """When no existing deadline is found, a new one plus a timeline event must be created."""
        from database import CaseDeadline, CaseTimeline
        from case_manager import _auto_create_deadlines_from_remedies

        remedies = {"appeal_days": "15", "appeal_court": "District Court"}

        _auto_create_deadlines_from_remedies(
            db=test_db,
            user_id=2,
            case_id=100,
            case_title="CASE-NEW",
            remedies=remedies,
            document_id=7,
        )

        deadlines = test_db.query(CaseDeadline).filter(
            CaseDeadline.case_id == 100,
        ).all()
        assert len(deadlines) == 1
        assert deadlines[0].deadline_type == "appeal"

        events = test_db.query(CaseTimeline).filter(
            CaseTimeline.case_id == 100,
            CaseTimeline.event_type == "deadline_created",
        ).all()
        assert len(events) == 1

