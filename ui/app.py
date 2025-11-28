import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Detector de Géneros", page_icon="🎵")

st.title("🎵 Clasificador de Géneros Musicales")
st.write("Pega la letra de una canción y la IA detectará sus géneros.")

# Área de texto
lyrics = st.text_area("Letra de la canción:", height=300)

if st.button("Analizar Género", type="primary"):
    if not lyrics.strip():
        st.warning("Por favor, escribe algo de texto.")
    else:
        with st.spinner("Analizando ritmos y patrones..."):
            try:
                # Llamada a TU API local
                response = requests.post("http://127.0.0.1:8000/predict", json={"lyrics": lyrics})
                
                if response.status_code == 200:
                    data = response.json()
                    generos = data["genres"]
                    
                    if not generos:
                        st.info("No se detectó ningún género con suficiente confianza.")
                    else:
                        st.success("¡Análisis completado!")
                        # Mostrar resultados con barras de progreso
                        for item in generos:
                            label = item['label'].title()
                            score = item['score']
                            st.write(f"**{label}**")
                            st.progress(score)
                            st.caption(f"Confianza: {score:.1%}")
                else:
                    st.error("Error en la API. Asegúrate de que el backend está corriendo.")
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API. ¿Ejecutaste `uv run uvicorn...`?")