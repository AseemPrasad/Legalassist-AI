"""
Reminder Insights - LegalAssist AI.
Tracks reminder effectiveness and drop-off after notification delivery.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# Add parent directory to sys.path to resolve project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import get_current_user_id, redirect_to_login, require_auth
from analytics_engine import ReminderInsightsEngine
from database import SessionLocal


st.set_page_config(
    page_title="Reminder Insights - LegalAssist AI",
    page_icon="📣",
    layout="wide",
)

st.markdown(
    """
    <style>
        .insights-hero {
            padding: 24px 28px;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #1f2937 100%);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.20);
            margin-bottom: 1.5rem;
        }
        .insights-hero h1 { margin: 0; font-size: 2.1rem; }
        .insights-hero p { margin: 0.5rem 0 0; color: rgba(248,250,252,0.82); }
        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.78));
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        .section-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_metric(label: str, value: str, help_text: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color: #475569; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">{label}</div>
            <div style="margin-top: 0.35rem; font-size: 1.8rem; font-weight: 800; color: {accent};">{value}</div>
            <div style="margin-top: 0.35rem; color: #64748b; font-size: 0.84rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_breakdown_chart(title: str, frame: pd.DataFrame, group_col: str, color: str) -> None:
    st.markdown(f"<span class='section-chip'>{title}</span>", unsafe_allow_html=True)
    if frame.empty:
        st.info(f"No reminder data available for {title.lower()}.")
        return

    chart_frame = frame.head(10).copy()
    fig = px.bar(
        chart_frame,
        x=group_col,
        y="effectiveness_rate",
        color="drop_off_rate",
        color_continuous_scale=["#fde68a", "#f97316", "#b91c1c"],
        text="effectiveness_rate",
        hover_data={
            "reminders": True,
            "effective_reminders": True,
            "drop_off_reminders": True,
            "avg_days_to_completion": True,
            "effectiveness_rate": True,
            "drop_off_rate": True,
        },
        labels={
            group_col: title,
            "effectiveness_rate": "Effectiveness %",
            "drop_off_rate": "Drop-off %",
        },
        title=None,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Drop-off %"),
        xaxis_title=None,
        yaxis_title="Effectiveness %",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(chart_frame, use_container_width=True, hide_index=True)


if not require_auth():
    redirect_to_login()
    st.stop()

user_id = get_current_user_id()

st.markdown(
    """
    <div class="insights-hero">
        <h1>📣 Reminder Insights</h1>
        <p>Effectiveness is measured using the latest reminder sent before a deadline is completed. Deadlines that are still future-dated are excluded from drop-off counts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

attribution_window_days = st.slider(
    "Attribution window (days)",
    min_value=1,
    max_value=30,
    value=14,
    help="A reminder is considered effective if the deadline is completed within this many days after the send time.",
)

st.caption("Court is shown only when stored on the deadline record; older rows may appear as Not specified.")

with SessionLocal() as db:
    insights = ReminderInsightsEngine.build_insights(
        db=db,
        attribution_window_days=attribution_window_days,
        user_id=user_id,
    )

summary = insights["summary"]
frame = insights["frame"]
by_jurisdiction = insights["by_jurisdiction"]
by_court = insights["by_court"]
by_deadline_type = insights["by_deadline_type"]
by_channel = insights["by_channel"]

if st.button("Refresh insights", use_container_width=False):
    st.rerun()

metric_cols = st.columns(5)
with metric_cols[0]:
    _render_metric("Attributed reminders", str(summary["reminder_count"]), "Resolved reminders included in the selected attribution window.", "#0f766e")
with metric_cols[1]:
    _render_metric("Effective", str(summary["effective_reminders"]), "Attributed reminders followed by a completion event in-window.", "#15803d")
with metric_cols[2]:
    _render_metric("Drop-off", str(summary["drop_off_reminders"]), "Matured deadlines without an effective reminder outcome.", "#b45309")
with metric_cols[3]:
    _render_metric("Effectiveness", f"{summary['effectiveness_rate']:.1f}%", "Share of attributed reminders that led to completion.", "#1d4ed8")
with metric_cols[4]:
    avg_days = "-" if summary["avg_days_to_completion"] is None else f"{summary['avg_days_to_completion']:.2f}"
    _render_metric("Avg completion lag", avg_days, "Average days between the attributed reminder and completion.", "#7c3aed")

st.markdown("---")

if frame.empty:
    st.info("No reminder analytics data is available for your account yet.")
else:
    tab_jurisdiction, tab_court, tab_deadline_type, tab_channel, tab_details = st.tabs(
        ["By Jurisdiction", "By Court", "By Deadline Type", "By Channel", "Details"]
    )

    with tab_jurisdiction:
        _render_breakdown_chart("Jurisdiction", by_jurisdiction, "jurisdiction", "effectiveness_rate")

    with tab_court:
        _render_breakdown_chart("Court", by_court, "court_name", "effectiveness_rate")

    with tab_deadline_type:
        _render_breakdown_chart("Deadline Type", by_deadline_type, "deadline_type", "effectiveness_rate")

    with tab_channel:
        _render_breakdown_chart("Channel", by_channel, "channel", "effectiveness_rate")

    with tab_details:
        detail_frame = frame.copy().sort_values(["sent_at", "deadline_id"], ascending=[False, False])
        detail_frame = detail_frame[[
            "deadline_id",
            "notification_log_id",
            "jurisdiction",
            "court_name",
            "deadline_type",
            "channel",
            "sent_at",
            "completion_at",
            "effective",
            "drop_off",
            "days_to_completion",
        ]]
        st.dataframe(detail_frame.head(50), use_container_width=True, hide_index=True)

        csv_bytes = detail_frame.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download reminder insights CSV",
            data=csv_bytes,
            file_name=f"reminder_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
