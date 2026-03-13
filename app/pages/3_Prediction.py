import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from auth import require_auth

require_auth()

st.set_page_config(
    page_title="Application ML - Prediction",
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
.pred-box {
    background: linear-gradient(135deg, #e8f5e9, #dff3ff);
    padding: 24px;
    border-radius: 18px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    color: #103b2d;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-top: 12px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title(" Navigation")
st.sidebar.info("Page 3 : saisie des données et prédiction")

st.title(" Interface de Prédiction")
st.markdown(
    '<div class="small-text">Renseigne les variables d’entrée pour obtenir une prédiction du modèle entraîné.</div>',
    unsafe_allow_html=True
)

if "model" not in st.session_state:
    st.warning(" Veuillez d'abord entraîner un modèle dans la page Training.")
    st.stop()

if "df" not in st.session_state:
    st.warning(" Veuillez d'abord charger les données dans la page Data.")
    st.stop()

df = st.session_state["df"].copy()
model = st.session_state["model"]
target_col = st.session_state["target_col"]
X_columns = st.session_state["X_columns"]
label_encoder = st.session_state["label_encoder"]

feature_df = df.drop(columns=[target_col])

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Saisir les valeurs</div>', unsafe_allow_html=True)

input_data = {}
cols = st.columns(2)

for i, col in enumerate(feature_df.columns):
    with cols[i % 2]:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            min_val = float(feature_df[col].min())
            max_val = float(feature_df[col].max())
            mean_val = float(feature_df[col].mean())

            input_data[col] = st.number_input(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
        else:
            options = feature_df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(col, options)

st.markdown('</div>', unsafe_allow_html=True)

if st.button(" Prédire"):
    input_df = pd.DataFrame([input_data])

    input_encoded = pd.get_dummies(input_df)

    for col in X_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[X_columns]

    prediction = model.predict(input_encoded)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.markdown(
        f'<div class="pred-box">Classe prédite : {predicted_label}</div>',
        unsafe_allow_html=True
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_encoded)[0]
        prob_df = pd.DataFrame({
            "Classe": label_encoder.classes_,
            "Probabilité": probabilities
        })

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probabilités par classe</div>', unsafe_allow_html=True)
            st.dataframe(prob_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"> Visualisation des probabilités</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(prob_df["Classe"], prob_df["Probabilité"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilité")
            ax.set_title("Probabilités de prédiction", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)