import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title("🌸 Iris Flower Classifier")

# 1. Charger dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Entraîner modèle
model = RandomForestClassifier()
model.fit(X, y)

# 3. Interface avec sliders
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

features = [[sepal_length, sepal_width, petal_length, petal_width]]

# Bouton prédire
if st.button("Prédire"):

    prediction = model.predict(features)
    proba = model.predict_proba(features)

    species = iris.target_names[prediction][0]

    st.subheader("Espèce prédite :")
    st.success(species)

    # Probabilités
    st.subheader("Probabilités par classe")

    prob_df = pd.DataFrame(
        proba,
        columns=iris.target_names
    )

    st.bar_chart(prob_df.T)