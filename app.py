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
# CONFIG
# -------------------------
st.set_page_config(page_title="Cyberbullying Detector", layout="centered")

st.title("🚨 Detector de Ciberacoso")
st.write("Clasifica si un texto contiene ciberacoso usando NLP.")

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

# 🔥 balanceo básico
extra = pd.DataFrame({
    "tweet_text": [
        "I love you",
        "I love my family",
        "You are amazing",
        "This is a great day",
        "I enjoy learning",
        "You did a great job"
    ],
    "cyberbullying_type": ["not_cyberbullying"] * 6
})

df = pd.concat([df, extra], ignore_index=True)

# -------------------------
# ENTRENAMIENTO
# -------------------------
@st.cache_resource
def entrenar(df):

    df["clean"] = df["tweet_text"].apply(limpiar_texto)

    X = df["clean"]
    y = df["cyberbullying_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = model.score(vectorizer.transform(X_test), y_test)

    return model, vectorizer, accuracy

lr_final, tfidf_vectorizer, accuracy = entrenar(df)

# -------------------------
# PREDICCIÓN
# -------------------------
def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    vector = tfidf_vectorizer.transform([texto_limpio])

    pred = lr_final.predict(vector)[0]
    probs = lr_final.predict_proba(vector)[0]

    clases = lr_final.classes_
    idx = list(clases).index(pred)
    confianza = probs[idx]

    return pred, confianza

# -------------------------
# INTERFAZ
# -------------------------
texto = st.text_area("✍️ Escribe un tweet:")

if st.button("Analizar"):

    if texto.strip() == "":
        st.warning("Por favor ingresa un texto")
    else:
        resultado, confianza = predecir(texto)

        # 🔥 RESULTADO CLARO
        if resultado == "not_cyberbullying":
            st.success("✅ No es ciberacoso")
        else:
            st.error("🚨 Contenido ofensivo detectado")

        # 🔥 MÉTRICA DINÁMICA
        st.metric("Confianza del modelo", f"{confianza*100:.2f}%")

# -------------------------
# MÉTRICA GLOBAL
# -------------------------
st.subheader("📊 Desempeño del modelo")
st.metric("Accuracy", f"{accuracy:.2%}")
