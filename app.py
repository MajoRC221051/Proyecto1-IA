import streamlit as st
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Cyberbullying Detector", layout="centered")

st.title("🚨 Detector de Cyberbullying")

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
# DATOS
# -------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("cyberbullying_tweets.csv")

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

modelo, vectorizer, base_accuracy = entrenar_modelo(df)

# -------------------------
# SESSION STATE
# -------------------------
if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.correct = 0
    st.session_state.last_pred = None

# -------------------------
# INPUT
# -------------------------
texto_usuario = st.text_area("Escribe texto:")

if st.button("Predecir"):
    if texto_usuario.strip():
        limpio = limpiar_texto(texto_usuario)
        vector = vectorizer.transform([limpio])
        pred = modelo.predict(vector)[0]

        st.session_state.last_pred = pred

        st.success(f"Predicción: {pred}")

# -------------------------
# FEEDBACK DEL USUARIO
# -------------------------
if st.session_state.last_pred is not None:
    st.write("¿Fue correcta la predicción?")

    col1, col2 = st.columns(2)

    if col1.button("✅ Correcta"):
        st.session_state.total += 1
        st.session_state.correct += 1

    if col2.button("❌ Incorrecta"):
        st.session_state.total += 1

# -------------------------
# ACCURACY DINÁMICO
# -------------------------
st.subheader("📊 Accuracy dinámico")

if st.session_state.total > 0:
    dynamic_acc = st.session_state.correct / st.session_state.total
    st.metric("Accuracy en uso", f"{dynamic_acc:.2%}")
else:
    st.metric("Accuracy inicial (test)", f"{base_accuracy:.2%}")

# -------------------------
# CLASES
# -------------------------
st.bar_chart(df["cyberbullying_type"].value_counts())
