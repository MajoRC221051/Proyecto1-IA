import streamlit as st
import re
import nltk
import pandas as pd
import time

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
    df = pd.read_csv("cyberbullying_tweets.csv")
    return df

df = cargar_datos()

# -------------------------
# ENTRENAMIENTO DINÁMICO
# -------------------------
def entrenar_modelo_dinamico(df):
    df["clean_text"] = df["tweet_text"].apply(limpiar_texto)

    X = df["clean_text"]
    y = df["cyberbullying_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modelo con warm_start
    modelo = LogisticRegression(max_iter=1, warm_start=True)

    # UI dinámica
    progress_bar = st.progress(0)
    accuracy_placeholder = st.empty()

    epochs = 20
    acc_list = []

    for i in range(epochs):
        modelo.fit(X_train_vec, y_train)

        acc = modelo.score(X_test_vec, y_test)
        acc_list.append(acc)

        progress_bar.progress((i + 1) / epochs)
        accuracy_placeholder.metric("Accuracy en entrenamiento", f"{acc:.2%}")

        time.sleep(0.2)  # solo para visual

    return modelo, vectorizer, acc_list[-1]

# Botón para entrenar
if st.button("🚀 Entrenar modelo"):
    modelo, vectorizer, accuracy = entrenar_modelo_dinamico(df)
    st.session_state["modelo"] = modelo
    st.session_state["vectorizer"] = vectorizer
    st.session_state["accuracy"] = accuracy

# -------------------------
# INTERFAZ
# -------------------------
st.subheader("✍️ Ingresa un texto")

texto_usuario = st.text_area("Escribe aquí:", height=150)

if st.button("Predecir"):
    if "modelo" not in st.session_state:
        st.warning("Primero entrena el modelo 👆")
    elif texto_usuario.strip() != "":
        limpio = limpiar_texto(texto_usuario)
        vector = st.session_state["vectorizer"].transform([limpio])
        pred = st.session_state["modelo"].predict(vector)[0]

        st.success(f"Predicción: **{pred}**")

        with st.expander("🔍 Ver proceso"):
            st.write("**Texto original:**", texto_usuario)
            st.write("**Texto limpio:**", limpio)
    else:
        st.warning("Por favor ingresa un texto.")

# -------------------------
# MÉTRICAS
# -------------------------
st.subheader("📊 Desempeño del modelo")

if "accuracy" in st.session_state:
    st.metric(label="Accuracy final", value=f"{st.session_state['accuracy']:.2%}")
else:
    st.info("Entrena el modelo para ver métricas.")

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
    3. Entrenamiento iterativo (visualización en vivo)
    4. Clasificación con Regresión Logística
    """)
