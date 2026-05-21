"""
Example: Streamlit integration with REST API
"""
import streamlit as st
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk.python.client import LegalassistClient

st.set_page_config(
    page_title="Legalassist AI - Web & API",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legalassist AI - REST API Integration")

# Initialize client
@st.cache_resource
def get_client():
    api_key = st.secrets.get("API_KEY", "demo-key")
    return LegalassistClient(
        base_url="http://localhost:8000",
        api_key=api_key
    )

client = get_client()

# Sidebar navigation
page = st.sidebar.radio("Select Page", [
    "📄 Document Analysis",
    "🔍 Case Search",
    "📊 Reports",
    "⏰ Deadlines",
    "💰 Analytics",
    "🔐 Account"
])

# ============================================================================
# Document Analysis Page
# ============================================================================

if page == "📄 Document Analysis":
    st.header("Document Analysis via API")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Submit Document")
        document_text = st.text_area(
            "Paste document text or contract",
            height=300,
            placeholder="Enter legal document text here..."
        )
        
        doc_type = st.selectbox(
            "Document Type",
            ["contract", "lawsuit", "agreement", "other"]
        )
        
        if st.button("🚀 Analyze Document", type="primary"):
            if document_text:
                with st.spinner("Submitting analysis job..."):
                    result = client.analyze_document(
                        text=document_text,
                        document_type=doc_type
                    )
                    
                    st.session_state.job_id = result['job_id']
                    st.success(f"Job submitted! ID: {result['job_id']}")
            else:
                st.error("Please enter document text")
    
    with col2:
        st.subheader("Job Status")
        
        if "job_id" in st.session_state:
            job_id = st.session_state.job_id
            st.info(f"Current Job: {job_id}")
            
            if st.button("🔄 Check Status"):
                status = client.get_analysis_status(job_id)
                st.write(f"Status: **{status['status']}**")
                
                if status['status'] == 'completed':
                    st.success("✅ Analysis complete!")
                    
                    if st.button("📥 Load Results"):
                        result = client.get_analysis_result(job_id)
                        st.session_state.analysis_result = result
        else:
            st.info("No active jobs")
    
    # Display results
    if "analysis_result" in st.session_state:
        st.subheader("📋 Analysis Results")
        result = st.session_state.analysis_result
        
        # Summary
        st.markdown("### Summary")
        st.write(result['summary'])
        
        # Key Points
        if result.get('key_points'):
            st.markdown("### Key Points")
            for point in result['key_points']:
                st.write(f"• {point}")
        
        # Remedies
        if result.get('remedies'):
            st.markdown("### Available Remedies")
            for remedy in result['remedies']:
                with st.expander(f"{remedy['type'].title()} - {remedy['jurisdiction']}"):
                    st.write(remedy['description'])

        remedies_confidence = result.get('remedies_confidence_score')
        if remedies_confidence is not None:
            st.metric("Remedies confidence", f"{remedies_confidence * 100:.1f}%")

        evidence_spans = result.get('remedies_evidence_spans') or []
        if evidence_spans:
            with st.expander("Remedies evidence excerpts"):
                for span in evidence_spans:
                    st.markdown(f"**{span.get('field', 'field')}**")
                    st.caption(span.get('snippet_reason', 'Evidence excerpt'))
                    st.write(span.get('span_text', ''))
        
        # Deadlines
        if result.get('deadlines'):
            st.markdown("### Extracted Deadlines")
            for deadline in result['deadlines']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{deadline['title']}**")
                    st.write(deadline['description'])
                with col2:
                    st.write(f"📅 {deadline['date'][:10]}")
        
        # Confidence
        st.markdown("---")
        confidence = result['confidence_score']
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Analysis Confidence")
        with col2:
            st.metric("Score", f"{confidence*100:.1f}%")


# ============================================================================
# Case Search Page
# ============================================================================

elif page == "🔍 Case Search":
    st.header("Search Similar Cases via API")
    
    with st.form("case_search"):
        col1, col2 = st.columns(2)
        
        with col1:
            keywords = st.text_input(
                "Keywords",
                placeholder="Enter search keywords"
            )
            jurisdiction = st.selectbox(
                "Jurisdiction",
                ["US", "UK", "CA", "AU"]
            )
        
        with col2:
            case_type = st.selectbox(
                "Case Type",
                ["civil", "criminal", "contract", "labor"]
            )
            limit = st.slider("Results Limit", 5, 50, 10)
        
        if st.form_submit_button("🔍 Search Cases", type="primary"):
            if keywords:
                with st.spinner("Searching cases..."):
                    results = client.search_cases(
                        keywords=keywords.split(),
                        jurisdiction=jurisdiction,
                        limit=limit
                    )
                    
                    st.session_state.search_results = results
                    st.success(f"Found {results['total_results']} cases")
            else:
                st.error("Please enter keywords")
    
    if "search_results" in st.session_state:
        results = st.session_state.search_results
        
        st.subheader(f"Results ({len(results['results'])} shown)")
        
        for case in results['results']:
            with st.expander(f"📋 {case['title']} ({case['case_number']})"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Case Number:** {case['case_number']}")
                    st.write(f"**Year:** {case['year']}")
                    st.write(f"**Jurisdiction:** {case['jurisdiction']}")
                    st.write(f"**Summary:** {case['summary']}")
                    st.write(f"**Verdict:** {case['verdict']}")
                
                with col2:
                    st.metric("Relevance", f"{case['relevance_score']*100:.0f}%")
                
                with col3:
                    if st.button("📅 Timeline", key=case['case_id']):
                        st.session_state.selected_case = case['case_id']


# ============================================================================
# Reports Page
# ============================================================================

elif page == "📊 Reports":
    st.header("Generate Reports via API")
    
    with st.form("report_form"):
        case_id = st.text_input("Case ID", placeholder="Enter case ID")
        report_type = st.selectbox(
            "Report Type",
            ["comprehensive", "summary", "legal_brief"]
        )
        report_format = st.selectbox("Format", ["pdf", "docx", "html"])
        
        col1, col2 = st.columns(2)
        with col1:
            include_remedies = st.checkbox("Include Remedies", value=True)
        with col2:
            include_timeline = st.checkbox("Include Timeline", value=True)
        
        if st.form_submit_button("📄 Generate Report", type="primary"):
            if case_id:
                with st.spinner("Generating report..."):
                    result = client.generate_report(
                        case_id=case_id,
                        report_type=report_type,
                        format=report_format
                    )
                    
                    st.session_state.report_job_id = result['job_id']
                    st.success(f"Report generation started: {result['job_id']}")
            else:
                st.error("Please enter case ID")
    
    if "report_job_id" in st.session_state:
        st.subheader("Report Status")
        job_id = st.session_state.report_job_id
        
        if st.button("📊 Check Report Status"):
            status = client.get_report_status(job_id)
            st.write(f"Status: **{status['status']}**")
            
            if status['status'] == 'completed' and status.get('download_url'):
                st.success("✅ Report ready!")
                st.write(f"Download: {status['download_url']}")


# ============================================================================
# Deadlines Page
# ============================================================================

elif page == "⏰ Deadlines":
    st.header("Upcoming Deadlines via API")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        days = st.slider("Look ahead (days)", 7, 90, 30)
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    with st.spinner("Loading deadlines..."):
        deadlines = client.get_upcoming_deadlines(days=days)
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Deadlines", deadlines['total_deadlines'])
    with col2:
        st.metric("🔴 Critical", deadlines['critical_count'])
    with col3:
        st.metric("🟠 High", deadlines['high_count'])
    with col4:
        st.metric("🟡 Medium", deadlines['medium_count'])
    
    # Deadlines list
    for deadline in deadlines['deadlines']:
        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(deadline['priority'], "⚪")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.write(f"{priority_emoji} **{deadline['title']}**")
            st.write(f"📅 Due: {deadline['due_date'][:10]} ({deadline['days_until_due']} days)")
            st.write(f"_{deadline['description']}_")
        
        with col2:
            if deadline['days_until_due'] <= 3:
                st.error(f"{deadline['days_until_due']}d")
            elif deadline['days_until_due'] <= 7:
                st.warning(f"{deadline['days_until_due']}d")
            else:
                st.info(f"{deadline['days_until_due']}d")


# ============================================================================
# Analytics Page
# ============================================================================

elif page == "💰 Analytics":
    st.header("API Usage & Costs")
    
    with st.spinner("Loading analytics..."):
        costs = client.get_cost_breakdown(period="monthly")
        overview = client.get_analytics_overview()
    
    # Cost breakdown
    st.subheader("Monthly Costs")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${costs['cost_breakdown']['total_cost']:.2f}")
    with col2:
        st.metric("LLM API", f"${costs['cost_breakdown']['llm_api_cost']:.2f}")
    with col3:
        st.metric("Processing", f"${costs['cost_breakdown']['document_processing_cost']:.2f}")
    with col4:
        st.metric("Storage", f"${costs['cost_breakdown']['storage_cost']:.2f}")
    
    # Usage metrics
    st.subheader("Usage Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Calls", costs['cost_breakdown']['api_calls'])
    with col2:
        st.metric("Documents", costs['cost_breakdown']['documents_analyzed'])
    with col3:
        st.metric("Reports", costs['cost_breakdown']['reports_generated'])


# ============================================================================
# Account Page
# ============================================================================

elif page == "🔐 Account":
    st.header("Account & API Keys")
    
    with st.spinner("Loading account info..."):
        user = client.get_current_user()
    
    st.subheader("Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**User ID:** {user['user_id']}")
        st.write(f"**Email:** {user['email']}")
    
    with col2:
        st.write(f"**Role:** {user['role']}")
        st.write(f"**Tier:** {user.get('subscription_tier', 'Free')}")
    
    # API Keys
    st.subheader("API Keys")
    
    with st.form("create_api_key"):
        key_name = st.text_input("Key Name", placeholder="e.g., Production")
        expires_days = st.number_input("Expires in (days)", min_value=1, max_value=365, value=90)
        
        if st.form_submit_button("🔑 Create API Key"):
            new_key = client.create_api_key(key_name, expires_days)
            st.success("API Key created!")
            st.code(new_key['key'], language="text")
            st.warning("⚠️ Save this key in a secure location. You won't see it again!")
    
    st.info("ℹ️ Manage your API keys from the /docs endpoint")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ## API Documentation
    - [Interactive Docs](/docs)
    - [API Reference](https://github.com/legalassist-ai/docs)
    - [SDK Examples](/examples)
    """
)
