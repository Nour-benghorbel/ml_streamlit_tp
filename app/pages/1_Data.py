import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from auth import require_auth

require_auth()



st.set_page_config(
    page_title="Application ML - Data",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #f6f8fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #1e3a5f;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    font-size: 28px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 8px;
}
.small-text {
    color: #5b657a;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title(" Navigation")
st.sidebar.info("Page 1 : importation et exploration des données")

st.title(" Upload et Exploration des Données")
st.markdown(
    '<div class="small-text">Charge un fichier CSV puis explore rapidement sa structure, ses statistiques et ses distributions.</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df

    st.success(" Fichier chargé avec succès")

    total_missing = int(df.isnull().sum().sum())
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lignes", df.shape[0])
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        st.metric("Colonnes numériques", len(numeric_cols))
    with col4:
        st.metric("Valeurs manquantes", total_missing)

    st.markdown("---")

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👀 Aperçu du dataset</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"> Informations générales</div>', unsafe_allow_html=True)
        st.write(f"**Nombre de lignes :** {df.shape[0]}")
        st.write(f"**Nombre de colonnes :** {df.shape[1]}")
        st.write(f"**Colonnes catégorielles :** {len(categorical_cols)}")
        st.write("**Noms des colonnes :**")
        st.write(list(df.columns))
        st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"> Types des colonnes</div>', unsafe_allow_html=True)
        dtypes_df = pd.DataFrame({
            "Colonne": df.columns,
            "Type": df.dtypes.astype(str).values
        })
        st.dataframe(dtypes_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"> Valeurs manquantes</div>', unsafe_allow_html=True)
        missing_df = pd.DataFrame({
            "Colonne": df.columns,
            "Valeurs manquantes": df.isnull().sum().values
        }).sort_values(by="Valeurs manquantes", ascending=False)
        st.dataframe(missing_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Statistiques descriptives</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if len(numeric_cols) > 0:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"> Visualisation d\'une variable numérique</div>', unsafe_allow_html=True)

        selected_col = st.selectbox("Choisir une colonne numérique", numeric_cols)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(df[selected_col].dropna(), bins=20, edgecolor="black")
        ax.set_title(f"Distribution de {selected_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Fréquence")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("📥 Veuillez uploader un fichier CSV pour commencer l'exploration.")