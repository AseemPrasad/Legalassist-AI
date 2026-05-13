import streamlit as st
from database import SessionLocal, get_organization_members, get_organization_audit_logs, create_organization, add_member_to_organization, UserRole, get_user_by_email, create_user
from auth import require_auth, get_current_user_id, get_current_user_email
import pandas as pd

# Set page config
st.set_page_config(page_title="Team Management - LegalEase AI", page_icon="👥", layout="wide")

def main():
    if not require_auth():
        st.warning("Please log in to access team management.")
        if st.button("Go to Login"):
            st.switch_page("pages/0_Login.py")
        return

    user_id = get_current_user_id()
    org_id = st.session_state.get("org_id")
    user_role = st.session_state.get("user_role")

    st.title("👥 Team Management")

    if not org_id:
        st.info("You are not currently part of an organization.")
        if st.button("🚀 Create Organization / Law Firm"):
            st.session_state.show_create_org = True
        
        if st.session_state.get("show_create_org"):
            with st.form("create_org_form"):
                org_name = st.text_input("Organization Name", placeholder="e.g. Justice & Co. Law Firm")
                submitted = st.form_submit_button("Create")
                if submitted and org_name:
                    db = SessionLocal()
                    org = create_organization(db, org_name)
                    add_member_to_organization(db, user_id, org.id, UserRole.ADMIN)
                    db.close()
                    st.session_state.org_id = org.id
                    st.session_state.org_name = org.name
                    st.session_state.user_role = UserRole.ADMIN
                    st.success(f"Organization '{org_name}' created successfully!")
                    st.rerun()
        return

    st.sidebar.markdown(f"### 🏢 {st.session_state.org_name}")
    st.sidebar.markdown(f"**Role:** {user_role.value.capitalize() if user_role else 'N/A'}")

    tabs = st.tabs(["👥 Team Members", "📜 Audit Logs", "🎨 Branding & Settings"])

    with tabs[0]:
        st.header("Team Members")
        db = SessionLocal()
        members = get_organization_members(db, org_id)
        
        member_data = []
        for m in members:
            member_data.append({
                "Email": m.user.email,
                "Role": m.role.value.capitalize(),
                "Joined": m.joined_at.strftime("%Y-%m-%d")
            })
        
        st.table(pd.DataFrame(member_data))

        if user_role == UserRole.ADMIN:
            st.markdown("---")
            st.subheader("➕ Add New Member")
            with st.form("add_member_form"):
                new_email = st.text_input("User Email")
                new_role = st.selectbox("Role", [UserRole.LAWYER, UserRole.PARALEGAL])
                add_submitted = st.form_submit_button("Add to Team")
                
                if add_submitted and new_email:
                    target_user = get_user_by_email(db, new_email)
                    if not target_user:
                        target_user = create_user(db, new_email)
                    
                    try:
                        add_member_to_organization(db, target_user.id, org_id, new_role)
                        st.success(f"Added {new_email} to the team as {new_role.value}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"User is already a member or error occurred: {str(e)}")
        db.close()

    with tabs[1]:
        st.header("Organization Audit Logs")
        if user_role != UserRole.ADMIN:
            st.error("Only Admins can view audit logs.")
        else:
            db = SessionLocal()
            logs = get_organization_audit_logs(db, org_id)
            if logs:
                log_data = []
                for l in logs:
                    log_data.append({
                        "Timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "User": l.user.email if l.user else "System",
                        "Action": l.action,
                        "Resource": f"{l.resource_type} ({l.resource_id})" if l.resource_type else "N/A",
                        "Details": str(l.details) if l.details else ""
                    })
                st.dataframe(pd.DataFrame(log_data), use_container_width=True)
            else:
                st.info("No audit logs found.")
            db.close()

    with tabs[2]:
        st.header("White-Labeling & Branding")
        if user_role != UserRole.ADMIN:
            st.info("Branding settings are read-only for your role.")
            # Show current settings (mockup for now)
            st.write("**Primary Color:** #2d2dff")
            st.write("**Logo URL:** Not set")
        else:
            with st.form("branding_form"):
                primary_color = st.color_picker("Primary Brand Color", "#2d2dff")
                logo_url = st.text_input("Logo URL", placeholder="https://example.com/logo.png")
                save_branding = st.form_submit_button("Save Branding")
                if save_branding:
                    # In a real app, update Organization.branding_settings in DB
                    st.success("Branding settings saved! (Simulated)")

if __name__ == "__main__":
    main()
