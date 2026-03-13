import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Application ML - Training",
    page_icon="🧠",
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

st.sidebar.title("📌 Navigation")
st.sidebar.info("Page 2 : entraînement et évaluation du modèle")

st.title("🧠 Entraînement du Modèle")
st.markdown(
    '<div class="small-text">Choisis la variable cible, sélectionne un modèle et visualise ses performances.</div>',
    unsafe_allow_html=True
)

if "df" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord charger un dataset dans la page Data.")
    st.stop()

df = st.session_state["df"].copy()

col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    target_col = st.selectbox("Choisir la colonne cible (target)", df.columns)

    model_name = st.selectbox(
        "Choisir le modèle",
        ["Logistic Regression", "Random Forest"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👀 Aperçu du dataset</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Entraîner le modèle"):
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.nunique() < 2:
        st.error("❌ La colonne cible doit contenir au moins 2 classes différentes.")
        st.stop()

    class_counts = y.value_counts()
    if (class_counts < 2).any():
        st.error(
            "❌ Certaines classes ont moins de 2 occurrences. "
            "Choisis une autre colonne cible, par exemple `LoanApproved`."
        )
        st.stop()

    X = pd.get_dummies(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    st.session_state["target_col"] = target_col
    st.session_state["X_columns"] = X.columns.tolist()
    st.session_state["label_encoder"] = label_encoder

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        split_msg = "Split stratifié utilisé"
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42
        )
        split_msg = "Split classique utilisé"

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    with st.spinner("Entraînement du modèle en cours..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    st.session_state["model"] = model

    st.success("✅ Modèle entraîné avec succès")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Accuracy", f"{acc:.2%}")
    with col_m2:
        st.metric("Nb classes", y.nunique())
    with col_m3:
        st.metric("Nb features", X.shape[1])
    with col_m4:
        st.metric("Échantillons", len(df))

    st.info(f"ℹ️ {split_msg}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📌 Vérification de la cible</div>', unsafe_allow_html=True)
        st.write(f"**Colonne cible choisie :** {target_col}")
        class_dist_df = pd.DataFrame({
            "Classe": class_counts.index,
            "Occurrences": class_counts.values
        })
        st.dataframe(class_dist_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Matrice de confusion</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm)
    plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title("Confusion Matrix", pad=15, fontsize=14, fontweight="bold")
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Valeurs réelles")
    st.pyplot(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if model_name == "Random Forest":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌲 Importance des variables</div>', unsafe_allow_html=True)

        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importances, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        top_features = importances.head(10).sort_values(by="Importance", ascending=True)
        ax2.barh(top_features["Feature"], top_features["Importance"])
        ax2.set_title("Top 10 Features Importantes", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Importance")
        ax2.grid(axis="x", alpha=0.2)
        st.pyplot(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)