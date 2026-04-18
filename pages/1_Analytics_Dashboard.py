"""
Analytics Dashboard for LegalEase AI

Shows case success rates, patterns, judge performance, and regional trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import SessionLocal, get_db
from analytics_engine import (
    AnalyticsCalculator,
    AnalyticsAggregator,
    CaseSimilarityCalculator,
)
import logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Analytics Dashboard - LegalEase AI",
    page_icon="📊",
    layout="wide",
)

# Styling
st.markdown("""
<style>
    body {
        background-color: #0d0d0f;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d2dff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Analytics Dashboard")
st.markdown("*Track case outcomes, success rates, and appeal patterns*")
st.markdown("---")

# Get database session
db = SessionLocal()

try:
    # ==================== SUMMARY METRICS ====================
    st.subheader("📈 Overall Statistics")
    
    summary = AnalyticsAggregator.get_dashboard_summary(db)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📁 Total Cases",
            summary["total_cases_processed"],
            delta=None,
            help="Total cases processed in the system"
        )
    
    with col2:
        st.metric(
            "📤 Appeals Filed",
            summary["appeals_filed"],
            delta=f"{summary['appeal_rate_percent']:.1f}% of all cases",
        )
    
    with col3:
        st.metric(
            "🏆 Plaintiff Wins",
            summary["plaintiff_wins"],
            help="Cases where plaintiff/complainant won"
        )
    
    with col4:
        st.metric(
            "⚖️ Settlements",
            summary["settlements"],
            help="Cases settled out of court"
        )
    
    st.markdown("---")
    
    # ==================== CASE OUTCOME DISTRIBUTION ====================
    st.subheader("📊 Case Outcome Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of outcomes
        outcomes_data = {
            "Plaintiff Won": summary["plaintiff_wins"],
            "Defendant Won": summary["defendant_wins"],
            "Settlement": summary["settlements"],
            "Dismissal": summary["dismissals"],
        }
        
        # Filter out zeros
        outcomes_data = {k: v for k, v in outcomes_data.items() if v > 0}
        
        if outcomes_data:
            fig = px.pie(
                values=list(outcomes_data.values()),
                names=list(outcomes_data.keys()),
                title="Case Outcomes",
                color_discrete_sequence=["#2d2dff", "#8a2be2", "#00d4ff", "#ff006e"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#0d0d0f",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No case data available yet.")
    
    with col2:
        # Appeal filing trends
        if summary["total_cases_processed"] > 0:
            appeal_data = {
                "Appeals Filed": summary["appeals_filed"],
                "No Appeal": summary["total_cases_processed"] - summary["appeals_filed"],
            }
            
            fig = px.bar(
                x=list(appeal_data.keys()),
                y=list(appeal_data.values()),
                title="Appeal Filing Rate",
                labels={"x": "", "y": "Number of Cases"},
                color_discrete_sequence=["#2d2dff", "#666"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#0d0d0f",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== JURISDICTION SELECTION ====================
    st.subheader("🗺️ Regional Analysis")
    
    # Get all jurisdictions
    from database import CaseRecord
    all_cases = db.query(CaseRecord).all()
    jurisdictions = sorted(set(case.jurisdiction for case in all_cases if case.jurisdiction))
    
    if jurisdictions:
        selected_jurisdiction = st.selectbox("Select Jurisdiction", jurisdictions)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional statistics
            regional_stats = AnalyticsCalculator.calculate_jurisdiction_trends(
                db, selected_jurisdiction
            )
            
            st.subheader(f"📍 {selected_jurisdiction}")
            
            jur_cases = [case for case in all_cases if case.jurisdiction == selected_jurisdiction]
            appeal_rate = AnalyticsCalculator.calculate_appeal_success_rate(jur_cases)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Cases", regional_stats.get("total_cases", 0))
            with col_b:
                st.metric("Appeal Success Rate", f"{appeal_rate:.1f}%")
            
            # Case type breakdown
            if regional_stats.get("case_type_stats"):
                st.subheader("By Case Type")
                type_data = regional_stats["case_type_stats"]
                
                df_types = pd.DataFrame([
                    {
                        "Case Type": case_type,
                        "Count": stats["count"],
                        "Plaintiff Win Rate": f"{stats['plaintiff_win_rate']:.1f}%",
                    }
                    for case_type, stats in type_data.items()
                ])
                
                st.dataframe(df_types, use_container_width=True)
        
        with col2:
            # Judge analytics for jurisdiction
            st.subheader(f"👨‍⚖️ Top Judges in {selected_jurisdiction}")
            
            judges = AnalyticsAggregator.get_top_judges(db, selected_jurisdiction, limit=10)
            
            if judges:
                df_judges = pd.DataFrame(judges)
                df_judges = df_judges[[
                    "judge",
                    "total_cases",
                    "win_rate",
                    "appeal_success_rate"
                ]]
                df_judges.columns = [
                    "Judge",
                    "Cases",
                    "Win Rate %",
                    "Appeal Success %"
                ]
                
                st.dataframe(df_judges, use_container_width=True)
                
                # Visualization
                if judges:
                    fig = px.bar(
                        df_judges.head(5),
                        x="Judge",
                        y=["Win Rate %", "Appeal Success %"],
                        title="Top 5 Judges by Success Rate",
                        barmode="group",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0d0d0f",
                        plot_bgcolor="#0d0d0f",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No judge data available for this jurisdiction yet.")
    else:
        st.info("No case data available yet. Cases will appear here after being processed.")
    
    st.markdown("---")
    
    # ==================== NATIONAL TRENDS ====================
    st.subheader("🌍 National Trends")
    
    regional_trends = AnalyticsAggregator.get_regional_trends(db)
    
    if regional_trends:
        df_trends = pd.DataFrame(regional_trends)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data table
            st.subheader("By Jurisdiction")
            st.dataframe(
                df_trends.sort_values("total_cases", ascending=False),
                use_container_width=True
            )
        
        with col2:
            # Visualization
            fig = px.bar(
                df_trends.sort_values("appeal_success_rate", ascending=False),
                x="jurisdiction",
                y="appeal_success_rate",
                title="Appeal Success Rate by Jurisdiction",
                labels={
                    "jurisdiction": "Jurisdiction",
                    "appeal_success_rate": "Success Rate %"
                }
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#0d0d0f",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Regional data will appear as more cases are processed.")
    
    st.markdown("---")
    
    # ==================== INSIGHTS ====================
    st.subheader("💡 Key Insights")
    
    if summary["total_cases_processed"] > 0:
        insights = []
        
        if summary["appeal_rate_percent"] > 50:
            insights.append("🔴 High appeal rate detected - more cases are being appealed than usual.")
        elif summary["appeal_rate_percent"] < 20:
            insights.append("🟢 Low appeal rate - users are satisfied with outcomes.")
        else:
            insights.append("🟡 Moderate appeal rate - steady appeal activity.")
        
        if summary["plaintiff_wins"] > summary["defendant_wins"]:
            insights.append("📈 Plaintiffs are winning more cases than defendants.")
        else:
            insights.append("📉 Defendants are winning more cases than plaintiffs.")
        
        for insight in insights:
            st.info(insight)
    else:
        st.info("Insights will appear as more case data is collected.")

except Exception as e:
    logger.error(f"Dashboard error: {str(e)}")
    st.error(f"Error loading dashboard: {str(e)}")

finally:
    db.close()

# ==================== ABOUT ====================
st.markdown("---")
st.markdown("""
### About This Dashboard

This analytics dashboard aggregates anonymized case data to provide:
- **Outcome Tracking**: Monitor case success rates by type and jurisdiction
- **Judge Analytics**: See how specific judges perform on appeals
- **Regional Trends**: Compare appeal success rates across different courts
- **Informed Decisions**: Help users understand their appeal chances

All data is anonymized and aggregated to protect user privacy.
""")
