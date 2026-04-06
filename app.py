import streamlit as st
import joblib
import re

# ===============================
# LIMPIEZA DE TEXTO
# ===============================
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)      # eliminar URLs
    texto = re.sub(r"@\w+", "", texto)         # eliminar menciones
    texto = re.sub(r"#\w+", "", texto)         # eliminar hashtags
    texto = re.sub(r"[^a-zA-Záéíóúñ\s]", "", texto)  # caracteres especiales
    texto = texto.strip()
    return texto

# ===============================
# CARGAR MODELO
# ===============================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("modelo.pkl")
        return modelo
    except Exception as e:
        st.error("❌ Error cargando el modelo. Verifica que modelo.pkl exista.")
        st.stop()

modelo = cargar_modelo()

# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================
def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    return modelo.predict([texto_limpio])[0]

# ===============================
# CONFIGURACIÓN DE LA APP
# ===============================
st.set_page_config(
    page_title="Detector de Ciberacoso",
    page_icon="🚨",
    layout="centered"
)

# ===============================
# UI
# ===============================
st.title("🚨 Detector de Ciberacoso")
st.markdown("### Analiza texto y detecta contenido ofensivo usando IA")

st.write(
    "Este modelo analiza texto tipo tweet/comentario y determina si contiene ciberacoso."
)

# Input
texto = st.text_area(
    "✍️ Escribe aquí el texto:",
    height=150,
    placeholder="Ej: eres un inútil..."
)

# Botón
if st.button("🔍 Analizar"):
    if texto.strip() == "":
        st.warning("⚠️ Por favor ingresa un texto.")
    else:
        with st.spinner("Analizando..."):
            resultado = predecir(texto)

        # Resultado principal
        st.subheader("Resultado:")

        if resultado == "not_cyberbullying":
            st.success("✅ No es ciberacoso")
        else:
            st.error("⚠️ Posible ciberacoso detectado")

        # Mostrar etiqueta técnica
        with st.expander("Detalle técnico"):
            st.write(f"Predicción del modelo: `{resultado}`")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Proyecto de IA • Detección de Ciberacoso")
