"""
Analytics Dashboard for LegalEase AI

Shows case success rates, patterns, judge performance, and regional trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

from database import SessionLocal, get_db, CaseRecord, CaseOutcome
from analytics_engine import (
    AnalyticsCalculator,
    AnalyticsAggregator,
    CaseSimilarityCalculator,
)
import logging

logger = logging.getLogger(__name__)


def build_query_signature(
    jurisdiction: str,
    case_type: str,
    court_name: str,
    judge_name: str,
    year_from: int | None,
    year_to: int | None,
) -> str:
    """Create a stable signature for a similarity search."""
    parts = [
        f"jurisdiction={jurisdiction}",
        f"case_type={case_type}",
        f"court_name={court_name}",
        f"judge_name={judge_name}",
        f"year_from={year_from or ''}",
        f"year_to={year_to or ''}",
    ]
    return "|".join(parts)

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
    # ==================== MY TEAM'S PERFORMANCE ====================
    org_id = st.session_state.get("org_id")
    if org_id:
        st.subheader(f"🏢 My Team's Performance ({st.session_state.get('org_name')})")
        from database import get_organization_stats
        org_stats = get_organization_stats(db, org_id)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Team Cases", org_stats["total_cases"])
        with col2:
            st.metric("Active", org_stats["active_cases"])
        with col3:
            st.metric("Appealed", org_stats["appealed_cases"])
        with col4:
            st.metric("Closed", org_stats["closed_cases"])
        with col5:
            st.metric("Org Deadlines", org_stats["upcoming_deadlines"])
            
        st.markdown("---")
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

    # ==================== SIMILAR CASE SEARCH ====================
    st.subheader("🔍 Find Similar Cases")

    with st.expander("API settings", expanded=False):
        api_base_url = st.text_input("API base URL", value=st.session_state.get("api_base_url", "http://localhost:8000"))
        bearer_token = st.text_input("Bearer token", value=st.session_state.get("api_token", ""), type="password")
        st.session_state.api_base_url = api_base_url
        st.session_state.api_token = bearer_token

    col1, col2, col3 = st.columns(3)

    with col1:
        similarity_jurisdiction = st.selectbox("Jurisdiction", jurisdictions if jurisdictions else ["US"], key="similarity_jurisdiction")
        similarity_case_type = st.selectbox(
            "Case Type",
            ["general", "civil", "criminal", "contract", "family"],
            key="similarity_case_type",
        )

    with col2:
        similarity_court = st.text_input("Court name", key="similarity_court")
        similarity_judge = st.text_input("Judge name", key="similarity_judge")

    with col3:
        year_from = st.number_input("Year from", min_value=1900, max_value=2100, value=2020, step=1, key="similarity_year_from")
        year_to = st.number_input("Year to", min_value=1900, max_value=2100, value=datetime.now().year, step=1, key="similarity_year_to")

    relevance_threshold = st.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="The backend only returns cases with relevance_score strictly above this threshold.",
    )

    st.caption("Top 5 results are requested from POST /api/v1/cases/search with an optional court/judge/year filter set.")

    if st.button("Find Similar Cases", type="primary"):
        payload = {
            "jurisdiction": similarity_jurisdiction,
            "case_type": similarity_case_type,
            "court_name": similarity_court or None,
            "judge_name": similarity_judge or None,
            "year_from": int(year_from),
            "year_to": int(year_to),
            "limit": 5,
            "relevance_threshold": relevance_threshold,
            "query_signature": build_query_signature(
                similarity_jurisdiction,
                similarity_case_type,
                similarity_court,
                similarity_judge,
                int(year_from),
                int(year_to),
            ),
        }
        headers = {"Content-Type": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        try:
            response = requests.post(
                f"{api_base_url.rstrip('/')}/api/v1/cases/search",
                json=payload,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            st.session_state.similar_case_results = response.json()
            st.session_state.similar_case_query_signature = payload["query_signature"]
        except Exception as exc:
            st.error(f"Similarity search failed: {exc}")

    if "similar_case_results" in st.session_state:
        results_payload = st.session_state.similar_case_results
        similar_results = results_payload.get("results", [])

        if similar_results:
            st.metric(
                "Returned appeal success rate",
                f"{(results_payload.get('appeal_success_rate') or 0.0) * 100:.1f}%",
                help="Aggregate appeal success rate across the returned result set.",
            )
            st.metric("Returned appealed cases", results_payload.get("appealed_cases", 0))

            result_ids = [int(item["case_id"]) for item in similar_results if str(item.get("case_id", "")).isdigit()]
            appeal_lookup = {}
            if result_ids:
                outcome_rows = db.query(CaseOutcome).filter(CaseOutcome.case_id.in_(result_ids)).all()
                appeal_lookup = {
                    row.case_id: (row.appeal_filed, row.appeal_success)
                    for row in outcome_rows
                }

            table_rows = []
            for item in similar_results:
                case_id = int(item["case_id"])
                appeal_filed, appeal_success = appeal_lookup.get(case_id, (None, None))
                table_rows.append({
                    "Case": item.get("case_number"),
                    "Title": item.get("title"),
                    "Verdict": item.get("verdict"),
                    "Relevance": f"{item.get('relevance_score', 0.0) * 100:.1f}%",
                    "Appeal filed": "Yes" if appeal_filed else ("No" if appeal_filed is not None else "Unknown"),
                    "Appeal success": "Yes" if appeal_success else ("No" if appeal_success is not None else "Unknown"),
                })

            df_similar = pd.DataFrame(table_rows)
            st.dataframe(df_similar, use_container_width=True)

            for item in similar_results:
                with st.expander(f"{item.get('title')} ({item.get('case_number')})"):
                    st.write(f"**Verdict:** {item.get('verdict')}")
                    st.write(f"**Relevance score:** {item.get('relevance_score', 0.0):.4f}")
                    st.write(f"**Jurisdiction:** {item.get('jurisdiction')}")
                    st.write(f"**Case type:** {item.get('case_type')}")
                    if item.get("appeal_success_rate") is not None:
                        st.write(f"**Appeal success:** {item['appeal_success_rate'] * 100:.0f}%")
        else:
            st.info("No similar cases met the current threshold.")
    
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
    # ==================== MODEL OPTIMIZATION ====================
    st.subheader("🤖 Model Optimization & Feedback")

    if st.button("Refresh model performance", key="refresh_models"):
        try:
            api_base = st.session_state.get("api_base_url", "http://localhost:8000").rstrip('/')
            token = st.session_state.get("api_token", "")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            resp = requests.get(f"{api_base}/api/v1/models/performance", headers=headers, timeout=10)
            resp.raise_for_status()
            st.session_state.model_performance = resp.json()
        except Exception as e:
            st.error(f"Could not fetch model performance: {e}")

    if "model_performance" in st.session_state:
        perf = st.session_state.model_performance
        items = perf.get("items", [])
        if items:
            df = pd.DataFrame(items)
            df = df[["model_name", "task", "case_type", "jurisdiction", "samples", "accuracy"]]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No model feedback collected yet.")

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
