"""
CLI tool for managing deadlines and testing the notification system.
Useful for bulk operations and testing without Streamlit.

Usage:
    python deadline_cli.py add-deadline --user-id user1 --case-id CASE-001 \\
        --case-title "Property Dispute" --days 30 --type appeal
    
    python deadline_cli.py list-deadlines --user-id user1
    
    python deadline_cli.py send-reminders --daysefore 30
    
    python deadline_cli.py test-sms --user-id user1 --message "Test message"
    
    python deadline_cli.py db-init
"""

import click
import sys
import json
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging

from database import (
    SessionLocal,
    init_db,
    create_case_deadline,
    create_or_update_user_preference,
    get_user_deadlines,
    get_upcoming_deadlines,
    NotificationChannel,
)
from notification_service import NotificationService
from scheduler import check_reminders_sync, trigger_reminder_check_now

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

notification_service = NotificationService()


@click.group()
def cli():
    """LegalAssist AI - Deadline Management CLI"""
    pass


# ==================== Database Commands ====================

@cli.command()
def db_init():
    """Initialize database and create tables"""
    click.echo("🔧 Initializing database...")
    try:
        init_db()
        click.secho("✅ Database initialized successfully", fg="green")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)


@cli.command()
@click.option("--user-id", required=True, help="User ID")
@click.option("--email", required=True, help="Email address")
@click.option("--phone", required=False, help="Phone number (with country code)")
@click.option("--timezone", default="UTC", help="Timezone (e.g., Asia/Kolkata)")
@click.option("--channel", type=click.Choice(["sms", "email", "both"]), default="both")
def setup_preferences(user_id: str, email: str, phone: Optional[str], timezone: str, channel: str):
    """Set up user notification preferences"""
    db = SessionLocal()
    try:
        channel_enum = {
            "sms": NotificationChannel.SMS,
            "email": NotificationChannel.EMAIL,
            "both": NotificationChannel.BOTH,
        }[channel]

        pref = create_or_update_user_preference(
            db=db,
            user_id=user_id,
            email=email,
            phone_number=phone,
            notification_channel=channel_enum,
            timezone=timezone,
        )
        
        click.secho(f"✅ Preferences saved for {user_id}", fg="green")
        click.echo(f"   Email: {pref.email}")
        click.echo(f"   Phone: {pref.phone_number or 'Not set'}")
        click.echo(f"   Channel: {pref.notification_channel.value}")
        click.echo(f"   Timezone: {pref.timezone}")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


# ==================== Deadline Commands ====================

@cli.command()
@click.option("--user-id", required=True, help="User ID")
@click.option("--case-id", required=True, help="Case ID")
@click.option("--case-title", required=True, help="Case title")
@click.option("--days", type=int, default=30, help="Days until deadline (default 30)")
@click.option("--type", type=click.Choice(["appeal", "filing", "submission", "response", "hearing", "other"]), default="appeal")
@click.option("--description", help="Additional notes")
def add_deadline(user_id: str, case_id: str, case_title: str, days: int, type: str, description: Optional[str]):
    """Add a new case deadline"""
    db = SessionLocal()
    try:
        deadline_date = datetime.now(timezone.utc) + timedelta(days=days)
        
        deadline = create_case_deadline(
            db=db,
            user_id=user_id,
            case_id=case_id,
            case_title=case_title,
            deadline_date=deadline_date,
            deadline_type=type,
            description=description,
        )
        
        formatted_date = deadline.deadline_date.strftime("%d %B %Y")
        click.secho(f"✅ Deadline added:", fg="green")
        click.echo(f"   Case: {deadline.case_title} ({deadline.case_id})")
        click.echo(f"   Deadline: {formatted_date}")
        click.echo(f"   Days: {days}")
        click.echo(f"   Reminders: 30, 10, 3, 1 day(s) before")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.option("--user-id", required=True, help="User ID")
def list_deadlines(user_id: str):
    """List all deadlines for a user"""
    db = SessionLocal()
    try:
        deadlines = get_user_deadlines(db, user_id)
        
        if not deadlines:
            click.echo("No active deadlines found")
            return
        
        click.secho(f"\n📋 Deadlines for {user_id}:", fg="cyan", bold=True)
        click.echo("=" * 80)
        
        for deadline in deadlines:
            days_left = deadline.days_until_deadline()
            emoji = "🔴" if days_left <= 3 else "🟠" if days_left <= 10 else "🟢"
            
            click.echo(f"{emoji} {deadline.case_title} ({deadline.deadline_type.upper()})")
            click.echo(f"   Case ID: {deadline.case_id}")
            click.echo(f"   Deadline: {deadline.deadline_date.strftime('%d %B %Y')}")
            click.echo(f"   Days Left: {days_left}")
            if deadline.description:
                click.echo(f"   Notes: {deadline.description}")
            click.echo()
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.option("--days-before", type=int, default=30, help="Check deadlines X days before")
def list_upcoming(days_before: int):
    """List all upcoming deadlines in the system"""
    db = SessionLocal()
    try:
        deadlines = get_upcoming_deadlines(db, days_before=days_before)
        
        if not deadlines:
            click.echo(f"No deadlines in the next {days_before} days")
            return
        
        click.secho(f"\n📅 Upcoming Deadlines (next {days_before} days):", fg="cyan", bold=True)
        click.echo("=" * 100)
        
        for deadline in deadlines:
            days_left = deadline.days_until_deadline()
            click.echo(f"User: {deadline.user_id} | Case: {deadline.case_id} | Days Left: {days_left}")
        
        click.secho(f"\nTotal: {len(deadlines)} upcoming deadlines", fg="green")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


# ==================== Reminder Commands ====================

@cli.command()
@click.option("--days", type=int, default=30, help="Check reminders for X days away")
def send_reminders(days: int):
    """Manually trigger reminder check for specific day threshold"""
    click.echo(f"📬 Checking and sending reminders for {days}-day threshold...")
    with click.progressbar(length=100) as bar:
        bar.update(50)
        count = check_reminders_sync(target_days=days)
        bar.update(50)
    
    click.secho(f"✅ Complete: {count} reminders sent", fg="green")


@cli.command()
def check_all_reminders():
    """Manually trigger complete reminder check (all thresholds)"""
    click.echo("📬 Running complete reminder check...")
    try:
        trigger_reminder_check_now()
        click.secho("✅ Reminder check completed", fg="green")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)


# ==================== Testing Commands ====================

@cli.command()
@click.option("--user-id", required=True)
@click.option("--case-title", default="Test Case")
@click.option("--days-left", type=int, default=10)
def test_sms(user_id: str, case_title: str, days_left: int):
    """Test SMS sending"""
    db = SessionLocal()
    try:
        from database import UserPreference, CaseDeadline
        
        # Get user preferences
        pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not pref or not pref.phone_number:
            click.secho(f"❌ No phone number configured for {user_id}", fg="red")
            sys.exit(1)
        
        # Create a test deadline
        deadline = CaseDeadline(
            user_id=user_id,
            case_id="TEST-001",
            case_title=case_title,
            deadline_date=datetime.now(timezone.utc) + timedelta(days=days_left),
            deadline_type="test",
        )
        
        # Send test SMS
        click.echo(f"📱 Sending test SMS to {pref.phone_number}...")
        result = notification_service.send_sms_reminder(db, deadline, pref, days_left)
        
        if result.success:
            click.secho("✅ SMS sent successfully", fg="green")
            click.echo(f"   Message ID: {result.message_id}")
            click.echo(f"   Recipient: {result.recipient}")
        else:
            click.secho(f"❌ SMS failed: {result.error}", fg="red")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.option("--user-id", required=True)
@click.option("--case-title", default="Test Case")
@click.option("--days-left", type=int, default=10)
def test_email(user_id: str, case_title: str, days_left: int):
    """Test email sending"""
    db = SessionLocal()
    try:
        from database import UserPreference, CaseDeadline
        
        # Get user preferences
        pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not pref:
            click.secho(f"❌ No preferences configured for {user_id}", fg="red")
            sys.exit(1)
        
        # Create a test deadline
        deadline = CaseDeadline(
            user_id=user_id,
            case_id="TEST-001",
            case_title=case_title,
            deadline_date=datetime.now(timezone.utc) + timedelta(days=days_left),
            deadline_type="test",
        )
        
        # Send test email
        click.echo(f"📧 Sending test email to {pref.email}...")
        result = notification_service.send_email_reminder(db, deadline, pref, days_left)
        
        if result.success:
            click.secho("✅ Email sent successfully", fg="green")
            click.echo(f"   Message ID: {result.message_id}")
            click.echo(f"   Recipient: {result.recipient}")
        else:
            click.secho(f"❌ Email failed: {result.error}", fg="red")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


@cli.command()
def test_config():
    """Test if credentials are configured"""
    import os
    
    click.echo("🔍 Checking configuration...\n")
    
    checks = {
        "TWILIO_ACCOUNT_SID": "Twilio SMS",
        "SENDGRID_API_KEY": "SendGrid Email",
        "DATABASE_URL": "Database",
    }
    
    for env_var, service in checks.items():
        is_set = env_var in os.environ
        status = "✅" if is_set else "⚠️"
        click.echo(f"{status} {service}: {'Configured' if is_set else 'Not configured'}")
    
    click.echo("\n" + "="*50)
    
    # Test database connection
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        click.secho("✅ Database connection: OK", fg="green")
    except Exception as e:
        click.secho(f"❌ Database connection: FAILED - {str(e)}", fg="red")


# ==================== Stats Command ====================

@cli.command()
def stats():
    """Show system statistics"""
    db = SessionLocal()
    try:
        from database import CaseDeadline, UserPreference, NotificationLog
        
        total_deadlines = db.query(CaseDeadline).count()
        active_deadlines = db.query(CaseDeadline).filter(CaseDeadline.is_completed == False).count()
        total_users = db.query(UserPreference).count()
        total_notifications = db.query(NotificationLog).count()
        
        click.secho("\n📊 System Statistics", fg="cyan", bold=True)
        click.echo("=" * 50)
        click.echo(f"Total Deadlines: {total_deadlines}")
        click.echo(f"Active Deadlines: {active_deadlines}")
        click.echo(f"Total Users: {total_users}")
        click.echo(f"Notifications Sent: {total_notifications}")
        
        if total_users > 0:
            avg_per_user = total_deadlines / total_users
            click.echo(f"Avg Deadlines/User: {avg_per_user:.1f}")
    except Exception as e:
        click.secho(f"❌ Error: {str(e)}", fg="red")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    cli()
