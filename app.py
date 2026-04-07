import streamlit as st
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# NLTK setup
# -------------------------
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

# -------------------------
# LIMPIEZA
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
# CARGAR DATASET
# -------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("cyberbullying_tweets.csv")

df = cargar_datos()

# 🔥 agregar ejemplos positivos (importante)
extra = pd.DataFrame({
    "tweet_text": [
        "I love my family",
        "You are amazing",
        "This is a great day",
        "I enjoy learning",
        "You did a great job"
    ],
    "cyberbullying_type": ["not_cyberbullying"] * 5
})

df = pd.concat([df, extra], ignore_index=True)

# -------------------------
# ENTRENAR MODELO
# -------------------------
@st.cache_resource
def entrenar(df):

    df["clean"] = df["tweet_text"].apply(limpiar_texto)

    X = df["clean"]
    y = df["cyberbullying_type"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return model, vectorizer

lr_final, tfidf_vectorizer = entrenar(df)

# -------------------------
# FUNCIÓN DE PREDICCIÓN
# -------------------------
def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    vector = tfidf_vectorizer.transform([texto_limpio])
    return lr_final.predict(vector)[0]

# ===============================
# INTERFAZ
# ===============================

st.title("🚨 Detector de Ciberacoso")
st.write("Ingresa un tweet y el modelo predecirá si contiene ciberacoso.")

texto = st.text_area("Escribe aquí el tweet:")

if st.button("Predecir"):

    if texto.strip() == "":
        st.warning("Por favor ingresa un texto")

    else:
        resultado = predecir(texto)

        st.success(f"Predicción: {resultado}")

        # Mensaje más claro
        if resultado == "not_cyberbullying":
            st.success("✅ Contenido no ofensivo")
        else:
            st.error("🚨 Contenido potencialmente ofensivo")
