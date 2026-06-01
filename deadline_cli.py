"""
LegalAssist AI - Deadline Management & System Administration CLI
================================================================

This CLI tool provides robust management for legal deadlines, notification 
preferences, and system diagnostics. It is designed to be highly observable 
and reliable, ensuring non-zero exit codes on all failure paths.

Key Features:
- Comprehensive database health checks
- Detailed notification logging and diagnostics
- Bulk management of user preferences and deadlines
- Professional terminal UI with Rich integration
"""

import os
import sys
import json
import logging
import signal
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Any, Callable, TypeVar, Generator
from functools import wraps

import click
import pyfiglet
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.prompt import Confirm
from rich import box

from database import (
    SessionLocal,
    init_db,
    create_case_deadline,
    create_or_update_user_preference,
    get_user_deadlines,
    get_upcoming_deadlines,
    NotificationChannel,
    NotificationStatus,
    CaseDeadline,
    UserPreference,
    NotificationLog,
    User,
    CaseAnalytics,
)
from notification_service import NotificationService
from core.log_redaction import mask_email, mask_phone
from scheduler import check_reminders_sync, trigger_reminder_check_now

# ==============================================================================
# Configuration & Initialization
# ==============================================================================

# Setup Rich console
console = Console()

# Setup logging with Rich integration
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("deadline_cli")

# Initialize Notification Service
notification_service = NotificationService()

# ==============================================================================
# Error Handling & Context Managers
# ==============================================================================

def handle_errors(func: Callable) -> Callable:
    """
    Decorator to ensure consistent error handling across all CLI commands.
    Ensures a non-zero exit code on failure and provides rich error reporting.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            console.print(Panel(
                f"[bold red]Critical Error:[/bold red]\n{str(e)}",
                title="Execution Failed",
                border_style="red"
            ))
            logger.exception("CLI execution failed")
            sys.exit(1)
    return wrapper


class CLIContext:
    """Context manager for database sessions with transactional rollback on interrupt."""
    def __init__(self):
        self.db = SessionLocal()
        self._interrupted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            self._interrupted = True
            self.db.rollback()
            console.print("\n[yellow]⚠ Batch operation interrupted. Changes rolled back.[/yellow]")
            logger.warning("cli_interrupted", msg="Transaction rolled back due to interrupt")
            return True
        self.db.close()
        if exc_type:
            return False


T = TypeVar('T')


def batch_update_with_transaction(
    items: List[T],
    process_fn: Callable[[T, Any], None],
    batch_size: int = 50,
    db_session: Optional[Any] = None,
) -> Generator[T, None, None]:
    """
    Process items in batches with automatic transaction management.
    Handles KeyboardInterrupt gracefully by rolling back incomplete batches.

    Args:
        items: List of items to process
        process_fn: Function to call for each item (receives item and db session)
        batch_size: Number of items per transaction commit
        db_session: Optional existing database session

    Yields:
        Items as they are successfully processed
    """
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        for idx, item in enumerate(items):
            try:
                process_fn(item, db_session)
                db_session.commit()
                yield item

                if (idx + 1) % batch_size == 0:
                    logger.debug("batch_commit", processed=idx + 1, batch_size=batch_size)

            except KeyboardInterrupt:
                db_session.rollback()
                logger.warning(
                    "batch_interrupted",
                    processed=idx,
                    remaining=len(items) - idx - 1,
                    msg="Rolled back partial changes. Re-run to continue."
                )
                raise

            except Exception as e:
                db_session.rollback()
                logger.error("batch_item_failed", item_idx=idx, error=str(e))
                raise

    finally:
        if should_close:
            db_session.close()


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    console.print("\n[yellow]⚠ Received interrupt signal. Finishing current batch...[/yellow]")
    sys.exit(130)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==============================================================================
# CLI Group Definition
# ==============================================================================

@click.group()
def cli():
    """LegalAssist AI - Proactive Legal Deadline Management System"""
    ascii_banner = pyfiglet.figlet_format("LegalAssist", font="slant")
    console.print(f"[bold cyan]{ascii_banner}[/bold cyan]")
    console.print("[italic]Secure, Proactive, and Compliant Legal Deadline Tracking[/italic]\n")


# ==============================================================================
# Database & System Commands
# ==============================================================================

@cli.command(name="db-init")
@handle_errors
def db_init_command():
    """Initialize the database schema and default tables."""
    console.print("[bold yellow]🔧 Initializing system database...[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task(description="Creating tables...", total=None)
        init_db()
    
    console.print("[bold green]✅ Database schema initialized successfully.[/bold green]")


@cli.command(name="db-check")
@handle_errors
def db_check_command():
    """Perform a comprehensive database health check."""
    console.print("[bold yellow]🔍 Running database health diagnostics...[/bold yellow]")
    
    try:
        with CLIContext() as ctx:
            start_time = datetime.now()
            ctx.db.execute("SELECT 1")
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get some basic counts
            user_count = ctx.db.query(User).count()
            deadline_count = ctx.db.query(CaseDeadline).count()
            
            table = Table(title="Database Health Status", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Connection Status", "Online")
            table.add_row("Latency", f"{latency:.2f} ms")
            table.add_row("Total Registered Users", str(user_count))
            table.add_row("Active Deadlines", str(deadline_count))
            
            console.print(table)
            console.print("[bold green]✅ Database health check passed.[/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Database check FAILED:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command(name="test-config")
@handle_errors
def test_config():
    """Verify that all required environment variables and services are configured."""
    console.print("[bold yellow]🔍 Verifying system configuration...[/bold yellow]")
    
    checks = {
        "TWILIO_ACCOUNT_SID": "Twilio SMS Gateway",
        "TWILIO_AUTH_TOKEN": "Twilio Authentication",
        "SENDGRID_API_KEY": "SendGrid Email Service",
        "DATABASE_URL": "Primary Database",
        "JWT_SECRET": "Security Layer (JWT)",
    }
    
    results_table = Table(title="Configuration Audit", box=box.SIMPLE)
    results_table.add_column("Component", style="cyan")
    results_table.add_column("Status", justify="center")
    results_table.add_column("Details")
    
    failed = False
    for env_var, service in checks.items():
        is_set = env_var in os.environ
        if not is_set:
            failed = True
            status = "[bold red]MISSING[/bold red]"
            details = f"Required environment variable '{env_var}' is not defined."
        else:
            status = "[bold green]OK[/bold green]"
            details = "Correctly configured."
        
        results_table.add_row(service, status, details)
    
    console.print(results_table)
    
    # Test DB connection as well
    try:
        with CLIContext() as ctx:
            ctx.db.execute("SELECT 1")
            console.print("[bold green]✅ Database connection: OK[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Database connection: FAILED[/bold red] ({str(e)})")
        failed = True
    
    if failed:
        console.print("\n[bold red]❌ Configuration check failed. Some services may not function correctly.[/bold red]")
        sys.exit(1)
    else:
        console.print("\n[bold green]✨ All systems go! Configuration is valid.[/bold green]")


# ==============================================================================
# User & Preference Management
# ==============================================================================

@cli.command(name="user-setup")
@click.option("--user-id", required=True, type=int, help="User ID")
@click.option("--email", required=True, help="User email address")
@click.option("--phone", required=False, help="Phone number with country code")
@click.option("--tz", default="UTC", help="Timezone (e.g., Asia/Kolkata)")
@click.option("--channel", type=click.Choice(["sms", "email", "both"]), default="both")
@handle_errors
def setup_preferences(user_id: int, email: str, phone: Optional[str], tz: str, channel: str):
    """Set or update notification preferences for a user."""
    with CLIContext() as ctx:
        channel_enum = {
            "sms": NotificationChannel.SMS,
            "email": NotificationChannel.EMAIL,
            "both": NotificationChannel.BOTH,
        }[channel]

        pref = create_or_update_user_preference(
            db=ctx.db,
            user_id=user_id,
            email=email,
            phone_number=phone,
            notification_channel=channel_enum,
            timezone=tz,
        )
        
        console.print(Panel(
            f"ID: [bold]{user_id}[/bold]\n"
            f"Email: [cyan]{pref.email}[/cyan]\n"
            f"Phone: [cyan]{pref.phone_number or 'N/A'}[/cyan]\n"
            f"Channel: [magenta]{pref.notification_channel.value}[/magenta]\n"
            f"Timezone: [magenta]{pref.timezone}[/magenta]",
            title="✅ User Preferences Saved",
            border_style="green"
        ))


@cli.command(name="user-list")
@handle_errors
def list_users():
    """List all registered users and their preference status."""
    with CLIContext() as ctx:
        users = ctx.db.query(User).all()
        if not users:
            console.print("[yellow]No users found in the system.[/yellow]")
            return

        table = Table(title="System User Registry", box=box.DOUBLE_EDGE)
        table.add_column("ID", style="dim")
        table.add_column("Email", style="cyan")
        table.add_column("Verified", justify="center")
        table.add_column("Last Login", style="dim")
        table.add_column("Preferences", justify="center")

        for user in users:
            pref = ctx.db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
            pref_status = "[green]SET[/green]" if pref else "[red]MISSING[/red]"
            verified_status = "✅" if user.is_verified else "❌"
            last_login = user.last_login.strftime("%Y-%m-%d %H:%M") if user.last_login else "Never"
            
            table.add_row(
                str(user.id),
                user.email,
                verified_status,
                last_login,
                pref_status
            )
        
        console.print(table)


# ==============================================================================
# Deadline Management
# ==============================================================================

@cli.command(name="add-deadline")
@click.option("--user-id", required=True, type=int)
@click.option("--case-id", required=True, type=int)
@click.option("--case-title", required=True)
@click.option("--days", type=int, default=30)
@click.option("--type", type=click.Choice(["appeal", "filing", "submission", "response", "hearing", "other"]), default="appeal")
@click.option("--description", help="Additional notes")
@handle_errors
def add_deadline(user_id: int, case_id: int, case_title: str, days: int, type: str, description: Optional[str]):
    """Register a new case deadline."""
    with CLIContext() as ctx:
        deadline_date = datetime.now(timezone.utc) + timedelta(days=days)
        
        deadline = create_case_deadline(
            db=ctx.db,
            user_id=user_id,
            case_id=case_id,
            case_title=case_title,
            deadline_date=deadline_date,
            deadline_type=type,
            description=description,
        )
        
        formatted_date = deadline.deadline_date.strftime("%d %B %Y")
        console.print(Panel(
            f"Case: [bold]{deadline.case_title}[/bold] (ID: {deadline.case_id})\n"
            f"Deadline: [cyan]{formatted_date}[/cyan] ({days} days from now)\n"
            f"Type: [magenta]{type.upper()}[/magenta]\n"
            f"Reminders: Standard (30, 10, 3, 1 days before)",
            title="✅ Deadline Created",
            border_style="green"
        ))


@cli.command(name="list-deadlines")
@click.option("--user-id", required=True, type=int)
@handle_errors
def list_deadlines(user_id: int):
    """Display all active deadlines for a specific user."""
    with CLIContext() as ctx:
        deadlines = get_user_deadlines(ctx.db, user_id)
        
        if not deadlines:
            console.print(f"[yellow]No active deadlines found for User {user_id}.[/yellow]")
            return
        
        table = Table(title=f"Deadlines for User {user_id}", box=box.ROUNDED)
        table.add_column("Case Title", style="bold white")
        table.add_column("Type", style="dim")
        table.add_column("Due Date", style="cyan")
        table.add_column("Remaining", justify="right")
        table.add_column("Status")

        for d in deadlines:
            days_left = d.days_until_deadline()
            status_color = "red" if days_left <= 3 else "yellow" if days_left <= 10 else "green"
            status_emoji = "🚨" if days_left <= 3 else "⏳" if days_left <= 10 else "✅"
            
            table.add_row(
                d.case_title,
                d.deadline_type.upper(),
                d.deadline_date.strftime("%Y-%m-%d"),
                f"{days_left} days",
                f"[{status_color}]{status_emoji} ACTIVE[/{status_color}]"
            )
        
        console.print(table)


# ==============================================================================
# Notification Engine Commands
# ==============================================================================

@cli.command(name="send-reminders")
@click.option("--days", type=int, default=30, help="Check reminders for X days threshold")
@handle_errors
def send_reminders(days: int):
    """Trigger the manual reminder engine for a specific threshold."""
    console.print(f"[bold yellow]📬 Processing reminders for {days}-day threshold...[/bold yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(description="Scanning deadlines...", total=100)
        
        progress.update(task, advance=20, description="Connecting to database...")
        count = check_reminders_sync(target_days=days)
        progress.update(task, advance=80, description="Done!")
    
    console.print(f"[bold green]✨ Complete: {count} notification(s) dispatched.[/bold green]")


@cli.command(name="check-all")
@handle_errors
def check_all_reminders():
    """Trigger a full reminder scan across all system thresholds."""
    console.print("[bold yellow]📬 Initiating global reminder scan...[/bold yellow]")
    trigger_reminder_check_now()
    console.print("[bold green]✅ Global reminder check completed.[/bold green]")


@cli.command(name="logs-view")
@click.option("--limit", default=10, help="Number of logs to show")
@handle_errors
def view_logs(limit: int):
    """View recent notification delivery logs."""
    with CLIContext() as ctx:
        logs = ctx.db.query(NotificationLog).order_by(NotificationLog.created_at.desc()).limit(limit).all()
        
        if not logs:
            console.print("[yellow]No notification logs found.[/yellow]")
            return

        table = Table(title=f"Recent Notifications (Last {limit})", box=box.MINIMAL_HEAVY_HEAD)
        table.add_column("Sent At", style="dim")
        table.add_column("User", justify="center")
        table.add_column("Channel", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Recipient")
        table.add_column("Error", style="red")

        for log in logs:
            status_style = "green" if log.status == NotificationStatus.SENT else "red" if log.status == NotificationStatus.FAILED else "yellow"
            table.add_row(
                log.created_at.strftime("%H:%M:%S"),
                str(log.user_id),
                log.channel.value.upper(),
                f"[{status_style}]{log.status.value.upper()}[/{status_style}]",
                log.recipient,
                log.error_message or ""
            )
        
        console.print(table)


# ==============================================================================
# Testing & Diagnostics
# ==============================================================================

@cli.command(name="test-sms")
@click.option("--user-id", required=True, type=int)
@click.option("--msg", default="Test notification from LegalAssist CLI")
@handle_errors
def test_sms(user_id: int, msg: str):
    """Dispatch a test SMS to a configured user."""
    with CLIContext() as ctx:
        pref = ctx.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not pref or not pref.phone_number:
            console.print(f"[bold red]❌ User {user_id} has no phone number configured.[/bold red]")
            sys.exit(1)
        
        console.print(f"📱 Dispatching test SMS to [cyan]{mask_phone(pref.phone_number)}[/cyan]...")
        
        # Create a minimal test case if none exists for FK constraint
        test_case = ctx.db.query(Case).filter(Case.user_id == user_id).first()
        if not test_case:
            test_case = Case(user_id=user_id, case_number="TEST_CLI", case_type="test", jurisdiction="Test")
            ctx.db.add(test_case)
            ctx.db.commit()
        
        dummy_deadline = CaseDeadline(
            user_id=user_id,
            case_id=test_case.id,
            case_title="CLI Test Case",
            deadline_date=datetime.now(timezone.utc),
            deadline_type="test"
        )
        
        result = notification_service.send_sms_reminder(ctx.db, dummy_deadline, pref, 0)
        
        if result.success:
            console.print("[bold green]✅ SMS Sent Successfully.[/bold green]")
            console.print(f"   [dim]Message ID: {result.message_id}[/dim]")
        else:
            console.print(f"[bold red]❌ SMS Failed:[/bold red] {result.error}")
            sys.exit(1)


@cli.command(name="test-email")
@click.option("--user-id", required=True, type=int)
@handle_errors
def test_email(user_id: int):
    """Dispatch a test Email to a configured user."""
    with CLIContext() as ctx:
        pref = ctx.db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not pref:
            console.print(f"[bold red]❌ User {user_id} has no preferences configured.[/bold red]")
            sys.exit(1)
        
        console.print(f"📧 Dispatching test email to [cyan]{mask_email(pref.email)}[/cyan]...")
        
        # Create a minimal test case if none exists for FK constraint
        test_case = ctx.db.query(Case).filter(Case.user_id == user_id).first()
        if not test_case:
            test_case = Case(user_id=user_id, case_number="TEST_CLI", case_type="test", jurisdiction="Test")
            ctx.db.add(test_case)
            ctx.db.commit()
        
        dummy_deadline = CaseDeadline(
            user_id=user_id,
            case_id=test_case.id,
            case_title="CLI Test Case",
            deadline_date=datetime.now(timezone.utc),
            deadline_type="test"
        )
        
        result = notification_service.send_email_reminder(ctx.db, dummy_deadline, pref, 0)
        
        if result.success:
            console.print("[bold green]✅ Email Sent Successfully.[/bold green]")
            console.print(f"   [dim]Message ID: {result.message_id}[/dim]")
        else:
            console.print(f"[bold red]❌ Email Failed:[/bold red] {result.error}")
            sys.exit(1)


# ==============================================================================
# Analytics & Stats
# ==============================================================================

@cli.command(name="stats")
@handle_errors
def show_stats():
    """Display comprehensive system statistics."""
    with CLIContext() as ctx:
        total_deadlines = ctx.db.query(CaseDeadline).count()
active_deadlines = ctx.db.query(CaseDeadline).filter(CaseDeadline.is_completed.is_(False)).count()
        total_users = ctx.db.query(User).count()
        total_logs = ctx.db.query(NotificationLog).count()
        failed_logs = ctx.db.query(NotificationLog).filter(NotificationLog.status == NotificationStatus.FAILED).count()

        # Success rate calculation
        success_rate = 100.0
        if total_logs > 0:
            success_rate = ((total_logs - failed_logs) / total_logs) * 100

        console.print(Panel(
            f"Total Users: [bold cyan]{total_users}[/bold cyan]\n"
            f"Active Deadlines: [bold cyan]{active_deadlines}[/bold cyan] / {total_deadlines}\n"
            f"Notifications Dispatched: [bold cyan]{total_logs}[/bold cyan]\n"
            f"Delivery Success Rate: [bold {'green' if success_rate > 95 else 'yellow'}]{success_rate:.1f}%[/bold {'green' if success_rate > 95 else 'yellow'}]",
            title="📊 System Statistics Summary",
            border_style="blue",
            expand=False
        ))

        # Show top users by deadline count
        from sqlalchemy import func
        top_users = ctx.db.query(
            CaseDeadline.user_id, 
            func.count(CaseDeadline.id).label('count')
        ).group_by(CaseDeadline.user_id).order_by(func.count(CaseDeadline.id).desc()).limit(5).all()

        if top_users:
            table = Table(title="Top Users by Deadlines", box=box.SIMPLE_HEAD)
            table.add_column("User ID", style="dim")
            table.add_column("Deadline Count", justify="right")
            for uid, count in top_users:
                table.add_row(str(uid), str(count))
            console.print(table)


if __name__ == "__main__":
    cli()
