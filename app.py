
import streamlit as st
import re
import nltk
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Descargar recursos
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Cyberbullying Detector", layout="centered")

st.title("🚨 Detector de Cyberbullying")
st.write("Clasifica texto en diferentes tipos de cyberbullying usando NLP.")

# -------------------------
# PREPROCESAMIENTO
# -------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)

    palabras = texto.split()
    palabras = [w for w in palabras if w not in stop_words]
    palabras = [lemmatizer.lemmatize(w) for w in palabras]

    return " ".join(palabras)

# -------------------------
# CARGA DE DATOS
# -------------------------
@st.cache_data
def cargar_datos():
    # IMPORTANTE: reemplaza con tu ruta local si no usas KaggleHub
    df = pd.read_csv("cyberbullying_tweets.csv")
    return df

df = cargar_datos()

# -------------------------
# ENTRENAMIENTO
# -------------------------
@st.cache_resource
def entrenar_modelo(df):
    df["clean_text"] = df["tweet_text"].apply(limpiar_texto)

    X = df["clean_text"]
    y = df["cyberbullying_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_vec, y_train)

    accuracy = modelo.score(X_test_vec, y_test)

    return modelo, vectorizer, accuracy

modelo, vectorizer, accuracy = entrenar_modelo(df)

# -------------------------
# INTERFAZ
# -------------------------
st.subheader("✍️ Ingresa un texto")

texto_usuario = st.text_area("Escribe aquí:", height=150)

if st.button("Predecir"):
    if texto_usuario.strip() != "":
        limpio = limpiar_texto(texto_usuario)
        vector = vectorizer.transform([limpio])
        pred = modelo.predict(vector)[0]

        st.success(f"Predicción: **{pred}**")

        with st.expander("🔍 Ver proceso"):
            st.write("**Texto original:**", texto_usuario)
            st.write("**Texto limpio:**", limpio)
    else:
        st.warning("Por favor ingresa un texto.")

# -------------------------
# DISTRIBUCIÓN DE CLASES
# -------------------------
st.subheader("📌 Distribución de clases")

conteo = df["cyberbullying_type"].value_counts()
st.bar_chart(conteo)

# -------------------------
# INFO
# -------------------------
with st.expander("ℹ️ Cómo funciona"):
    st.write("""
    1. Limpieza del texto (NLP)
    2. Vectorización con TF-IDF
    3. Clasificación con Regresión Logística
    """)
