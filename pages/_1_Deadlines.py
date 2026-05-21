"""
Case Deadlines page - Manage appeal deadlines (stub redirects to notifications).
"""

import streamlit as st

st.set_page_config(
    page_title="Case Deadlines",
    page_icon="📅",
    layout="wide"
)

st.markdown("**This feature is available in the Deadline Tracker page.**")
if st.button("Go to Deadline Tracker"):
    st.switch_page("pages/3_Deadline_Tracker.py")