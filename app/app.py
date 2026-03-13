import streamlit as st

st.set_page_config(
    page_title="Application ML Multi-Pages",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- AUTH ----------
def check_login(username, password):
    try:
        return (
            username == st.secrets["auth"]["username"]
            and password == st.secrets["auth"]["password"]
        )
    except Exception:
        return False

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.title("🔐 Connexion")
    st.write("Authentifie-toi pour accéder à l'application.")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        if check_login(username, password):
            st.session_state.authenticated = True
            st.success("Connexion réussie")
            st.rerun()
        else:
            st.error("Identifiants incorrects")

if not st.session_state.authenticated:
    login()
    st.stop()

# Bouton logout dans la sidebar
if st.sidebar.button("🚪 Déconnexion"):
    st.session_state.authenticated = False
    st.rerun()

# ---------- STYLE ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #F8FAFC !important;
}
p, div, span {
    color: #E2E8F0;
}
.card {
    background-color: #1E293B;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    margin-bottom: 18px;
    min-height: 170px;
}
.card h3 {
    color: #FFFFFF !important;
    margin-bottom: 12px;
}
.card p {
    color: #CBD5E1 !important;
    font-size: 15px;
    line-height: 1.6;
}
.small-text {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 10px;
}
[data-testid="stSidebar"] {
    background-color: #1E293B;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- APP ----------
st.sidebar.title("📌 Navigation")
st.sidebar.success("Utilise le menu à gauche pour accéder aux pages.")

st.title("🤖 Application Machine Learning Multi-Pages")
st.markdown(
    '<div class="small-text">Application complète pour charger un dataset, entraîner un modèle et réaliser des prédictions.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📂 Data</h3>
        <p>Importe un fichier CSV, explore ses colonnes, visualise les statistiques et les distributions.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 Training</h3>
        <p>Choisis la variable cible, entraîne un modèle de classification et évalue ses performances.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>🔮 Prediction</h3>
        <p>Saisis de nouvelles valeurs d’entrée et obtiens une prédiction accompagnée des probabilités.</p>
    </div>
    """, unsafe_allow_html=True)

st.info("Commence par la page **Data** pour charger ton fichier CSV.")