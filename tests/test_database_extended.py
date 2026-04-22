
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
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
    create_case,
    get_user_cases,
    get_case_by_id,
    get_case_by_number,
    update_case_status,
    delete_case,
    create_case_document,
)

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
        record = CaseRecord(case_id="C1", case_type="civil", jurisdiction="Delhi", outcome="won")
        assert "CaseRecord" in repr(record)
        
        # CaseOutcome
        outcome = CaseOutcome(case_id=1, appeal_filed=True, appeal_success=False)
        assert "CaseOutcome" in repr(outcome)
        
        # UserFeedback
        feedback = UserFeedback(user_id="U1", appeal_outcome="lost")
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
            test_db, "CASE_ID_1", "civil", "Delhi", 
            court_name="High Court", outcome="plaintiff_won"
        )
        assert record.case_id == "CASE_ID_1"
        
        retrieved = get_case_record(test_db, "CASE_ID_1")
        assert retrieved.id == record.id
        
        # Criteria search
        results = get_cases_by_criteria(test_db, case_type="civil", jurisdiction="Delhi")
        assert len(results) == 1
        
        results = get_cases_by_criteria(test_db, jurisdiction="Mumbai")
        assert len(results) == 0

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
            test_db, "U1", did_appeal=True, appeal_outcome="won", satisfaction_rating=5
        )
        assert feedback.user_id == "U1"
        
        history = get_user_feedback(test_db, "U1")
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
        doc = create_case_document(test_db, case.id, DocumentType.JUDGMENT, "Some content")
        assert doc.document_type == DocumentType.JUDGMENT
        
        # Delete
        success = delete_case(test_db, case.id)
        assert success == True
        assert get_case_by_id(test_db, case.id) is None

    def test_init_db(self):
        """Test database initialization (schema creation)"""
        # This just ensures metadata.create_all doesn't crash
        init_db()
