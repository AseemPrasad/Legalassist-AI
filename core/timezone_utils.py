"""
Timezone utilities for consistent datetime display across the application.

Provides helpers for timezone-aware datetime handling and user-local display.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import pytz


def get_user_timezone(tz_name: Optional[str] = None) -> pytz.BaseTzInfo:
    """Get timezone object from name or return UTC as default."""
    if tz_name:
        try:
            return pytz.timezone(tz_name)
        except pytz.UnknownTimeZoneError:
            pass
    return pytz.UTC


def utc_to_local(dt: Union[datetime, str], tz_name: Optional[str] = None) -> datetime:
    """
    Convert UTC datetime to user's local timezone.
    
    Args:
        dt: UTC datetime (aware or naive) or ISO string
        tz_name: Target timezone name (e.g., 'Asia/Kolkata', 'America/New_York')
    
    Returns:
        Timezone-aware datetime in local time
    """
    local_tz = get_user_timezone(tz_name)
    
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    return dt.astimezone(local_tz)


def format_local_timestamp(dt: Union[datetime, str], tz_name: Optional[str] = None, fmt: str = "%d %b %Y, %I:%M %p") -> str:
    """
    Format UTC datetime as user-local string.
    
    Args:
        dt: UTC datetime or ISO string
        tz_name: Target timezone (defaults to user's browser timezone)
        fmt: strftime format string
    
    Returns:
        Formatted timestamp string in user's local time
    """
    local_dt = utc_to_local(dt, tz_name)
    return local_dt.strftime(fmt)


def format_deadline_timestamp(dt: Union[datetime, str], tz_name: Optional[str] = None) -> str:
    """Format deadline timestamp with day name for clarity."""
    local_dt = utc_to_local(dt, tz_name)
    day_name = local_dt.strftime("%A")
    date_str = local_dt.strftime("%d %b %Y, %I:%M %p")
    return f"{day_name}, {date_str}"


def get_timezone_offset(tz_name: Optional[str] = None) -> str:
    """Get timezone offset string for display (e.g., '+05:30')."""
    local_tz = get_user_timezone(tz_name)
    now = datetime.now(local_tz)
    offset = now.strftime('%z')
    
    hours = offset[:3]
    minutes = offset[3:] if len(offset) > 3 else '00'
    return f"{hours}:{minutes}"


def render_timezone_selector() -> Optional[str]:
    """Render Streamlit timezone selector and return selected timezone."""
    import streamlit as st
    
    common_timezones = [
        ("Asia/Kolkata", "India (IST)"),
        ("America/New_York", "US East (EST)"),
        ("America/Los_Angeles", "US West (PST)"),
        ("Europe/London", "UK (GMT)"),
        ("Europe/Paris", "Europe (CET)"),
        ("Asia/Dubai", "UAE (GST)"),
        ("Asia/Singapore", "Singapore (SGT)"),
        ("Australia/Sydney", "Australia (AEST)"),
    ]
    
    options = ["Auto-detect"] + [name for _, name in common_timezones]
    tz_map = {name: tz for tz, name in common_timezones}
    tz_map["Auto-detect"] = None
    
    selected = st.selectbox("Timezone", options, index=0)
    return tz_map.get(selected)