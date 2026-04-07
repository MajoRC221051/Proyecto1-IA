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

st.title("🚨 Cyberbullying Detector")
st.write("Detecta si un texto contiene cyberbullying y su tipo.")

# -------------------------
# PREPROCESAMIENTO
# -------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# -------------------------
# REGLAS ANTI-FALLO
# -------------------------
bullying_keywords = [
    "stupid", "idiot", "dumb", "useless", "loser",
    "hate you", "nobody likes you", "disgusting",
    "ugly", "trash", "kill yourself"
]

def rule_based_detection(text):
    text = text.lower()
    for word in bullying_keywords:
        if word in text:
            return 1
    return 0

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cyberbullying_tweets.csv")

df = load_data()

# -------------------------
# DATA AUGMENTATION
# -------------------------
extra_data = pd.DataFrame({
    "tweet_text": [
        "I love spending time with my family",
        "You are doing great, keep going",
        "This is a beautiful day",
        "I enjoy learning new things",
        "You are very talented",
        "Let's work together as a team",
        "I appreciate your help",
        "Everything will be okay",
        "You did a fantastic job",
        "I am proud of you"
    ],
    "cyberbullying_type": ["not_cyberbullying"] * 10
})

df = pd.concat([df, extra_data], ignore_index=True)

# -------------------------
# ETIQUETA BINARIA
# -------------------------
df["is_bullying"] = df["cyberbullying_type"].apply(
    lambda x: 0 if "not" in str(x).lower() else 1
)

# -------------------------
# TRAIN MODELS
# -------------------------
@st.cache_resource
def train_models(df):

    df["clean_text"] = df["tweet_text"].apply(clean_text)

    X = df["clean_text"]
    y_binary = df["is_bullying"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model_binary = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_binary.fit(X_train_vec, y_train)

    # Modelo tipo
    df_bully = df[df["is_bullying"] == 1]

    X2 = df_bully["clean_text"]
    y2 = df_bully["cyberbullying_type"]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=42
    )

    model_type = LogisticRegression(max_iter=1000)
    model_type.fit(vectorizer.transform(X2_train), y2_train)

    return model_binary, model_type, vectorizer

model_binary, model_type, vectorizer = train_models(df)

# -------------------------
# UI
# -------------------------
st.subheader("✍️ Ingresa un texto")

user_input = st.text_area("Escribe aquí:", height=150)

if st.button("Analizar"):

    if user_input.strip() != "":

        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        # 🔥 combinación modelo + reglas
        rule_pred = rule_based_detection(user_input)
        probs = model_binary.predict_proba(vector)[0]

        prob_no, prob_yes = probs

        pred_binary = 1 if (rule_pred == 1 or prob_yes > 0.5) else 0

        # 📊 GRÁFICA
        prob_df = pd.DataFrame({
            "Clase": ["No Bullying", "Bullying"],
            "Probabilidad": probs
        })

        st.subheader("📊 Probabilidad de predicción")
        st.bar_chart(prob_df.set_index("Clase"))

        # 🔥 MÉTRICAS DINÁMICAS
        st.metric("Riesgo de bullying", f"{prob_yes*100:.2f}%")
        st.metric("Confianza del modelo", f"{max(probs)*100:.2f}%")

        # RESULTADO
        if pred_binary == 0:
            st.success("✅ No es cyberbullying")
        else:
            st.error("🚨 Es cyberbullying")

            pred_type = model_type.predict(vector)[0]

            label_map = {
                "age": "Bullying por edad",
                "ethnicity": "Bullying por etnia",
                "religion": "Bullying por religión",
                "gender": "Bullying por género",
                "other_cyberbullying": "Otro tipo de bullying"
            }

            st.warning(f"Tipo: **{label_map.get(pred_type, pred_type)}**")

        with st.expander("🔍 Ver proceso"):
            st.write("Texto limpio:", cleaned)

    else:
        st.warning("Ingresa un texto.")
