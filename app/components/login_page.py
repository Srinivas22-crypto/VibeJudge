import streamlit as st
from auth.user_manager import UserManager
from auth.session_manager import session_manager

user_mgr = UserManager()

def render_login():
    """Render the login / registration page."""
    st.title("🎙️ VibeJudge")
    st.subheader("Batch Podcast Analyzer — Login")

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            result = user_mgr.authenticate_user(username, password)
            if result["success"]:
                token = session_manager.create_access_token(
                    {"sub": result["user_id"], "username": result["username"]}
                )
                st.session_state["token"] = token
                st.session_state["username"] = result["username"]
                st.session_state["logged_in"] = True
                st.success(f"Welcome back, {result['username']}! 🎉")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_name = st.text_input("Full Name (optional)")
            new_pass = st.text_input("Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Register", type="primary")

        if reg_submitted:
            if new_pass != confirm_pass:
                st.error("❌ Passwords do not match.")
            elif len(new_pass) < 6:
                st.error("❌ Password must be at least 6 characters.")
            else:
                result = user_mgr.create_user(new_user, new_email, new_pass, new_name)
                if result["success"]:
                    st.success("✅ Account created! Please login.")
                else:
                    st.error(f"❌ {result['error']}")

def is_logged_in() -> bool:
    """Check if user is authenticated."""
    if not st.session_state.get("logged_in"):
        return False
    token = st.session_state.get("token", "")
    result = session_manager.verify_token(token)
    return result["valid"]
