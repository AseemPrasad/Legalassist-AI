import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from database import (
    Base,
    CaseDeadline,
    UserPreference,
    NotificationChannel,
    create_case_deadline,
    create_or_update_user_preference,
    Case,
    CaseStatus,
    User,
)
from db.session import init_db
from notifications.reminder_engine import (
    get_reminder_dispatch_candidates,
    is_notify_enabled,
)
from api.main import create_app
from api.auth import CurrentUser, get_current_user

@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory test database with migrated tables"""
    engine = create_engine("sqlite:///:memory:")
    # Bind to session.py engine so init_db uses it
    with patch("db.session.engine", engine), patch("db.session._is_sqlite", True):
        init_db()
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

class TestCustomReminderThresholds:
    """Tests for user-configurable reminder thresholds"""

    def test_user_preference_custom_thresholds_persistence(self, test_db):
        """Test that custom thresholds are saved and retrieved correctly"""
        user = User(id=1, email="test@example.com")
        test_db.add(user)
        test_db.commit()

        # Create preferences with custom thresholds
        pref = create_or_update_user_preference(
            test_db,
            user_id=1,
            email="test@example.com",
            reminder_thresholds=[45, 15, 7]
        )

        assert pref.reminder_thresholds == [45, 15, 7]
        assert pref.get_reminder_thresholds() == [45, 15, 7]

        # Retrieve from db
        retrieved = test_db.query(UserPreference).filter_by(user_id=1).first()
        assert retrieved.reminder_thresholds == [45, 15, 7]
        assert retrieved.get_reminder_thresholds() == [45, 15, 7]

    def test_user_preference_fallback_thresholds(self, test_db):
        """Test fallback thresholds match legacy columns when reminder_thresholds is None"""
        user = User(id=1, email="test@example.com")
        test_db.add(user)
        test_db.commit()

        # Create preferences without custom thresholds
        pref = create_or_update_user_preference(
            test_db,
            user_id=1,
            email="test@example.com"
        )
        pref.reminder_thresholds = None
        pref.notify_30_days = True
        pref.notify_10_days = False
        pref.notify_3_days = True
        pref.notify_1_day = False
        test_db.commit()

        assert pref.get_reminder_thresholds() == [30, 3]

    def test_is_notify_enabled_custom_thresholds(self, test_db):
        """Test is_notify_enabled check against custom thresholds"""
        pref = UserPreference(reminder_thresholds=[45, 15, 7])
        assert is_notify_enabled(45, pref) is True
        assert is_notify_enabled(15, pref) is True
        assert is_notify_enabled(30, pref) is False

    def test_get_reminder_dispatch_candidates_custom_thresholds(self, test_db):
        """Test get_reminder_dispatch_candidates correctly fetches deadline for custom threshold"""
        now = datetime.now(timezone.utc)
        
        user = User(id=1, email="test@example.com")
        test_db.add(user)
        test_db.commit()

        case = Case(
            user_id=1,
            case_number="CASE-CUSTOM",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Custom Case",
        )
        test_db.add(case)
        test_db.commit()

        # Deadline 45 days away
        deadline = create_case_deadline(
            test_db,
            user_id=1,
            case_id=case.id,
            case_title="Custom Case",
            deadline_date=now + timedelta(days=45, hours=1),
            deadline_type="appeal",
        )

        pref = create_or_update_user_preference(
            test_db,
            user_id=1,
            email="test@example.com",
            reminder_thresholds=[45, 15, 7]
        )

        # Query candidates for 45 days
        candidates = get_reminder_dispatch_candidates(
            test_db,
            days_before=46,
            now_utc=now,
            reminder_time_checker=lambda tz: True,
        )

        assert len(candidates) == 1
        assert candidates[0][0].id == deadline.id
        assert candidates[0][1] == 45
        assert candidates[0][2].user_id == 1

    def test_api_preferences_endpoint(self, test_db):
        """Test GET and PUT /api/v1/notifications/preferences endpoints"""
        app = create_app()
        client = TestClient(app)

        # Mock current user and db
        mock_user = CurrentUser(user_id=1, email="api-test@example.com")
        
        app.dependency_overrides[get_current_user] = lambda: mock_user
        from api.dependencies import get_db_rls
        app.dependency_overrides[get_db_rls] = lambda: test_db

        user = User(id=1, email="api-test@example.com")
        test_db.add(user)
        test_db.commit()

        # 1. GET preferences (should create default preference)
        response = client.get("/api/v1/notifications/preferences")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert data["email"] == "api-test@example.com"
        assert data["reminder_thresholds"] == [30, 10, 3, 1]

        # 2. PUT preferences to custom thresholds
        payload = {
            "email": "new-email@example.com",
            "phone_number": "+1234567890",
            "notification_channel": "both",
            "timezone": "America/New_York",
            "reminder_thresholds": [45, 15, 7],
            "holiday_aware_reminders": True,
            "holiday_country": "US",
        }
        response = client.put("/api/v1/notifications/preferences", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "new-email@example.com"
        assert data["phone_number"] == "+1234567890"
        assert data["reminder_thresholds"] == [45, 15, 7]
        assert data["holiday_aware_reminders"] is True
        assert data["holiday_country"] == "US"

        # Cleanup overrides
        app.dependency_overrides.clear()
