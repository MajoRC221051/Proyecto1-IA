import streamlit as st

# IMPORTANTE: debes tener estas variables ya entrenadas
# limpiar_texto
# tfidf_vectorizer
# lr_final

# Función de predicción
def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    vector = tfidf_vectorizer.transform([texto_limpio])
    return lr_final.predict(vector)[0]

# ===============================
# INTERFAZ
# ===============================

st.title("Detector de Ciberacoso")
st.write("Ingresa un tweet y el modelo predirá si contiene ciberacoso.")

# Entrada de usuario
texto = st.text_area("Escribe aquí el tweet:")

# Botón
if st.button("Predecir"):
    if texto.strip() == "":
        st.warning("Por favor ingresa un texto")
    else:
        resultado = predecir(texto)

        st.success(f"Predicción: {resultado}")

        # Opcional: mostrar tipo
        if resultado == "not_cyberbullying":
            st.info("Contenido no ofensivo")
        else:
            st.error("Contenido potencialmente ofensivo")