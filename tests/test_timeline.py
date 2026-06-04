from database import init_db, SessionLocal, create_user, log_notification, User
from case_manager import create_new_case, upload_case_document, add_manual_deadline, get_case_full_timeline
from database import DocumentType
from datetime import datetime, timedelta, timezone


def test_case_full_timeline_ordering_and_linkage():
    # Ensure a fresh DB for this test (remove local sqlite DB if present)
    from pathlib import Path
    db_path = Path("./legalassist.db")
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    # Ensure tables exist
    init_db()

    db = SessionLocal()
    try:
        # Ensure any leftover test user is removed to keep tests idempotent
        try:
            db.query(User).filter(User.email == "timeline_test@example.com").delete(synchronize_session=False)
            db.commit()
        except Exception:
            db.rollback()
        # Create user
        user = create_user(db=db, email="timeline_test@example.com")
        user_id = user.id

        # Create case
        case, _ = create_new_case(user_id=user_id, case_number="TL/1/2026", case_type="civil", jurisdiction="TestLand", title="Timeline Test")
        case_id = case.id

        # Add a manual deadline
        deadline_date = datetime.now(timezone.utc) + timedelta(days=10)
        deadline = add_manual_deadline(user_id=user_id, case_id=case_id, case_title=case.title, deadline_date=deadline_date, deadline_type="filing", description="File response")
        assert deadline is not None
        deadline_id = deadline.id

        # Upload a document
        doc = upload_case_document(user_id=user_id, case_id=case_id, document_type=DocumentType.JUDGMENT, document_content="Sample text")
        assert doc is not None

        # Log a notification linked to the deadline
        n = log_notification(db=db, deadline_id=deadline_id, user_id=user_id, channel=__import__('database').NotificationChannel.SMS, recipient="+911234567890", days_before=10, status=__import__('database').NotificationStatus.SENT, message_id="m1", message_preview="Test SMS content")
        assert n is not None

        # Fetch unified timeline
        timeline = get_case_full_timeline(user_id=user_id, case_id=case_id)

        # There should be at least 4 items: case_created (from create_new_case), deadline_created, document_uploaded, reminder
        types = [it.get('type') for it in timeline]
        assert 'deadline_created' in types
        assert 'document_uploaded' in types
        assert 'reminder' in types or any('reminder' in (it.get('type') or '') for it in timeline)

        # Check that reminder item includes message preview and links to deadline id
        reminders = [it for it in timeline if it.get('type') == 'reminder']
        assert len(reminders) >= 1
        r = reminders[0]
        assert r.get('message_preview') == 'Test SMS content'
        assert r.get('metadata', {}).get('deadline_id') == deadline_id

    finally:
        db.close()
