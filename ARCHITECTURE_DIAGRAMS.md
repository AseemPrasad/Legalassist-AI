# Deadline Notification System - Architecture Diagrams

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LegalAssist AI Application                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │   Streamlit UI   │      │   CLI Tool       │               │
│  │  (UI Layer)      │      │  (Admin Tool)    │               │
│  │                  │      │                  │               │
│  │ • Deadlines      │      │ • Add deadlines  │               │
│  │ • Preferences    │      │ • Test SMS/Email │               │
│  │ • History        │      │ • View stats     │               │
│  └─────────┬────────┘      └─────────┬────────┘               │
│            │                          │                        │
│            └──────────────┬───────────┘                        │
│                           │                                    │
│        ┌──────────────────▼───────────────────┐               │
│        │    Application Layer                 │               │
│        │  (Business Logic)                    │               │
│        │                                      │               │
│        │  • Database operations               │               │
│        │  • Notification generation           │               │
│        │  • Preference management             │               │
│        └──────────────────┬───────────────────┘               │
│                           │                                    │
│        ┌──────────────────▼───────────────────┐               │
│        │    Data Layer                        │               │
│        │  (SQLAlchemy ORM)                    │               │
│        └──────────────────┬───────────────────┘               │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
       ┌────▼────┐   ┌──────▼─────┐   ┌───▼─────┐
       │ SQLite  │   │PostgreSQL  │   │ MySQL   │
       │(Dev)    │   │(Production)│   │(Alt)    │
       └─────────┘   └────────────┘   └─────────┘


        ┌─────────────────────────────────────┐
        │  Background Scheduler (APScheduler) │
        │  ▢ Daily at 8 AM UTC               │
        └──────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ Check Reminders Job │
        │                     │
        │ 1. Get deadlines   │
        │ 2. Build messages  │
        │ 3. Send via API    │
        │ 4. Log status      │
        └──────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
    ┌───▼──┐  ┌────▼─────┐  │
    │Twilio│  │ SendGrid  │  │
    │(SMS) │  │ (Email)   │  │
    └───────┘  └───────────┘  │
                              │
                    ┌─────────▼───────────┐
                    │  User Notification  │
                    │ SMS or Email        │
                    └─────────────────────┘
```

## Data Flow Diagram

```
Application Entry
        │
        ├─ Streamlit UI ──────┐
        │                      │
        ├─ CLI Tool ───────────┼─── User Input
        │                      │
        └─ API ────────────────┘
                    │
        ┌───────────▼────────────┐
        │  Validate & Process    │
        │  • Check credentials   │
        │  • Validate dates      │
        │  • Check permissions   │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Database Operations   │
        │  (SQLAlchemy ORM)      │
        │  • C: Create deadline  │
        │  • R: Get preferences  │
        │  • U: Update status    │
        │  • D: Delete if needed │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Notification Service  │
        │  • Check if should send│
        │  • Build message       │
        │  • Format for delivery │
        └───────────┬────────────┘
                    │
        ┌───────────┴─────────────────┐
        │                             │
    ┌───▼──────┐             ┌────────▼────┐
    │SMS Sender│             │Email Sender │
    │(Twilio)  │             │(SendGrid)   │
    └────┬─────┘             └───────┬─────┘
         │                           │
    ┌────▼──────────────────────────▼────┐
    │  External Service (API Call)       │
    │  • Send to Twilio/SendGrid API     │
    │  • Get response                     │
    │  • Capture message ID               │
    └────┬──────────────────────────┬────┘
         │                          │
    ┌────▼──────┐        ┌─────────▼────┐
    │ SMS Queue │        │ Email Queue   │
    │(External) │        │ (External)    │
    └────┬──────┘        └────────┬──────┘
         │                         │
    ┌────▼──────────────────────────▼───┐
    │  User Receives Notification       │
    │  • SMS on mobile                  │
    │  • Email in inbox                 │
    └──────────────────────────────────┘
         │                   │
         └──────────┬────────┘
                    │
    ┌───────────────▼──────────────┐
    │  Logging & Status Update     │
    │  • Mark as sent in DB        │
    │  • Store message ID          │
    │  • Log timestamp             │
    │  • Track delivery status     │
    └───────────────┬──────────────┘
                    │
    ┌───────────────▼──────────────┐
    │  User History                │
    │  • Viewable in UI            │
    │  • Sortable by date          │
    │  • Filterable by status      │
    └──────────────────────────────┘
```

## Reminder Timeline

```
Deadline Created: Day 0
    ↓
    │ Days pass...
    │
Day 60 Before: Scheduler checks ✓
    ├─ Not at reminder threshold (30/10/3/1)
    └─ Skip
    ↓
    │ Days pass...
    │
Day 30 Before: Scheduler checks ✓
    ├─ At reminder threshold = 30 days
    ├─ Get user preferences
    ├─ Check if already sent → NO
    ├─ Send SMS: "⚖️ LegalAssist: Your case deadline in 30 days!"
    ├─ Send Email: "⚖️ Urgent: Your case has a deadline in 30 days"
    ├─ Log both reminders as SENT
    └─ Store message IDs
    ↓
    │ Days pass...
    │
Day 20 Before: Scheduler checks ✓
    └─ Not at threshold, skip
    ↓
    │ Days pass...
    │
Day 10 Before: Scheduler checks ✓
    ├─ At reminder threshold = 10 days
    ├─ Check if already sent → NO (only 30-day was sent)
    ├─ Send SMS: "⚖️ LegalAssist: Your case deadline in 10 days!"
    ├─ Send Email: "⚖️ Urgent: Your case has a deadline in 10 days"
    ├─ Log both reminders as SENT
    └─ Store message IDs
    ↓
    │ Days pass...
    │
Day 3 Before: Scheduler checks ✓
    ├─ At reminder threshold = 3 days
    ├─ Send SMS: "🔴 CRITICAL: Your case deadline in 3 days!"
    ├─ Send Email: "🔴 CRITICAL: Deadline in 3 days - ACTION NEEDED"
    ├─ Log both reminders as SENT
    └─ Store message IDs
    ↓
    │ Days pass...
    │
Day 1 Before: Scheduler checks ✓
    ├─ At reminder threshold = 1 day
    ├─ Send SMS: "🚨 LAST CHANCE: Deadline TOMORROW!"
    ├─ Send Email: "🚨 LAST CHANCE: Deadline TOMORROW - Act now!"
    ├─ Log both reminders as SENT
    └─ Store message IDs
    ↓
Day 0: Deadline reached
    ├─ Scheduler still checks but no reminders sent
    └─ User marks as completed (or case moves to completed status)
```

## Database Relationships

```
┌─────────────────────────────┐
│   UserPreference            │
├─────────────────────────────┤
│ id (PK)                     │
│ user_id (UNIQUE, indexed)   │
│ email                       │
│ phone_number               │
│ notification_channel       │ ────┐
│ timezone                   │     │
│ notify_30_days             │     │
│ notify_10_days             │     │
│ notify_3_days              │     │
│ notify_1_day               │     │
│ created_at                 │     │
│ updated_at                 │     │
└─────────────────────────────┘     │
                                   │ (One-to-Many)
                                   │
┌─────────────────────────────┐     │
│   CaseDeadline              │     │
├─────────────────────────────┤     │
│ id (PK)                     │     │
│ user_id (indexed) ◄────────────┘
│ case_id                     │
│ case_title                  │
│ deadline_date (indexed)     │
│ deadline_type               │
│ description                 │
│ is_completed                │
│ created_at                  │
│ updated_at                  │
│ notifications (relation) ───┐
└─────────────────────────────┘ │
                                │ (One-to-Many)
                                │
┌──────────────────────────────┐│
│   NotificationLog            ││
├──────────────────────────────┤│
│ id (PK)                      ││
│ deadline_id (FK) ◄───────────┘
│ user_id (indexed)            │
│ channel (SMS/Email)          │
│ status                       │
│ recipient                    │
│ days_before (30/10/3/1)      │
│ message_id                   │
│ error_message                │
│ sent_at                      │
│ delivered_at                 │
│ created_at                   │
│ deadline (relation)          │
└──────────────────────────────┘
```

## System State Transitions

```
DEADLINE STATE:
    Created ──→ Active ──→ Notified (at 30 days)
        │                      │
        │                      ├──→ Notified (at 10 days)
        │                      │
        │                      ├──→ Notified (at 3 days)
        │                      │
        │                      ├──→ Notified (at 1 day)
        │                      │
        └──────────────→ Completed / Overdue

NOTIFICATION STATE (per deadline/threshold):
    Pending ──→ Sent ──→ Delivered
                 │
                 └──→ Failed ──→ Retry (optional)
                 
    User Views ──→ Opened (if email tracking enabled)
    User Bounced ──→ Bounced (if delivery failure)
```

## Scheduler Lifecycle

```
App Startup
    │
    ├─ Database Initialized
    │
    ├─ Scheduler Created
    │     └─ APScheduler with CronTrigger
    │
    ├─ Job Registered
    │     └─ check_and_send_reminders
    │        └─ Scheduled for: Daily at 8:00:00 UTC
    │
    └─ Scheduler Started
         └─ Running: True

During Execution (Daily):
    │
    ├─ Scheduler wakes up at 8 AM UTC
    │
    ├─ Job starts: check_and_send_reminders()
    │     ├─ Connect to database
    │     ├─ Query deadlines (30-day window)
    │     ├─ For each deadline on 30/10/3/1 day mark:
    │     │  ├─ Check user preferences
    │     │  ├─ Check if already sent (prevent duplicates)
    │     │  ├─ Build SMS/Email
    │     │  ├─ Call external APIs (Twilio/SendGrid)
    │     │  └─ Log result to database
    │     ├─ Close database connection
    │     └─ Job completes
    │
    ├─ Wait 24 hours
    │
    └─ Repeat tomorrow

App Shutdown:
    │
    └─ Scheduler stopped
         └─ Running: False
```

## Error Handling Flow

```
Notification Send Attempt
    │
    ├─ Try SMS/Email Send
    │
    ├─ Success? ──→ YES ──→ Status: SENT
    │                           │
    │                           └─→ Log message_id
    │
    ├─ Failure? ──→ YES ──→ Catch Exception
    │                           │
    │                           ├─ Log error_message
    │                           │
    │                           ├─ Status: FAILED
    │                           │
    │                           └─ Try Again? (Optional retry logic)
    │
    └─ Unknown? ──→ Status: FAILED, error_message = str(exception)

Error Types Handled:
    ├─ API Unavailable (Twilio/SendGrid down)
    ├─ Invalid Phone/Email
    ├─ Rate Limiting
    ├─ Network Timeout
    ├─ Database Errors
    └─ Invalid Configuration
```

---

## File Dependencies

```
app_integrated.py
    ├─ app.py (original)
    ├─ notifications_ui.py
    │   ├─ database.py
    │   ├─ notification_service.py
    │   │   ├─ database.py
    │   │   ├─ twilio (external)
    │   │   └─ sendgrid (external)
    │   ├─ scheduler.py
    │   │   ├─ database.py
    │   │   ├─ notification_service.py
    │   │   └─ apscheduler (external)
    │   └─ pytz (external)
    └─ scheduler.py

deadline_cli.py
    ├─ database.py
    ├─ notification_service.py
    ├─ scheduler.py
    └─ click (external)

tests/test_notifications.py
    ├─ database.py
    ├─ notification_service.py
    ├─ scheduler.py
    ├─ pytest (external)
    ├─ unittest.mock (stdlib)
    └─ sqlalchemy (external)
```

---

**These diagrams provide visual understanding of the system architecture, data flow, and state management.**
