"""
Settings/Preferences page - Notification preferences
"""

import streamlit as st

# Import the preferences UI
from notifications_ui import page_notification_preferences

st.set_page_config(
    page_title="Preferences",
    page_icon="⚙️",
    layout="wide"
)

if __name__ == "__main__":
    page_notification_preferences()
