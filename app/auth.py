import streamlit as st

def check_login(username, password):
    try:
        return (
            username == st.secrets["auth"]["username"]
            and password == st.secrets["auth"]["password"]
        )
    except Exception:
        return False

def require_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.warning("Veuillez vous connecter pour accéder à cette page.")
        st.stop()