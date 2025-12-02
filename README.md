# 🎵 Spotify Genre Classifier: End-to-End NLP Pipeline

Este proyecto desarrolla un flujo de trabajo completo (End-to-End MLOps) para la clasificación de géneros musicales utilizando Procesamiento de Lenguaje Natural (NLP). El sistema predice múltiples géneros para una canción basándose únicamente en su letra, utilizando modelos Transformer.

El enfoque principal de este repositorio es presentar una arquitectura de Deep Learning moderna y eficiente, desacoplando el entrenamiento pesado (en GPU) de la inferencia ligera (en CPU). El proyecto abarca desde la limpieza de texto y tokenización hasta el fine-tuning de DistilBERT con optimizaciones de memoria (FP16, Gradient Accumulation) y su despliegue como microservicio.

---

## 📋 Tabla de Contenidos
- [Arquitectura y Tech Stack](#-arquitectura-y-tech-stack)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Uso](#-instalación-y-uso)
- [Dashboard & API](#-dashboard--api)
- [Metodología de ML](#-metodología-de-ml)
- [Entrenamiento y resultados](#-entrenamiento-resultado)
- [Autor](#-autor)

---

## 🛠 Arquitectura y Tech Stack

El proyecto integra herramientas modernas para crear un sistema robusto, modular y escalable:

* **Lenguaje:** Python 3.11
* **Gestión de Dependencias:** [uv](https://github.com/astral-sh/uv) (Gestor de paquetes de alto rendimiento).
* **Modelado (NPL):** 
    * **Hugging Face Transformers:** Tokenización y arquitectura del modelo.
    * **PyTorch:** motor de cálculo tensorial.
    * **DistilBert:** modelo base (multilingual Cased) optimizado para eficiencia.
* **Infraestructura del modelo: Hugging Face Hub** (Alojamiento del modelo entrenado para mantener el repositorio ligero). 
* **Interfaces:** 
    * **FastAPI:** Backend para servir predicciones.
    * **Streamlit:** Frontend interactivo para el usuario final.
* **Entrenamiento:** Google Colab (T4 GPU) con estrategias de ahorro de memoria.

---

## 📂 Estructura del Proyecto

El código sigue una arquitectura de paquete modular, separando configuración, lógica y presentación:

```text
.
├── api/                 # 🔌 Backend (FastAPI)
│   ├── __init__.py
│   └── main.py          # Endpoints de la API
├── notebooks/           # 📓 Notebooks de Jupyter/Colab
│   └── training.ipynb   # Pipeline de entrenamiento completo
├── src/                 # 🧠 Lógica del Negocio
│   ├── __init__.py
│   ├── predictor.py     # Clase para descarga e inferencia del modelo
│   └── preprocessing.py # Limpieza y normalización de texto
├── ui/                  # 🎨 Frontend (Streamlit)
│   └── app.py           # Interfaz de usuario
├── .gitignore           # Archivos ignorados
├── pyproject.toml       # Definición de dependencias (uv)
├── uv.lock              # Versiones exactas (Lockfile)
└── README.md            # Documentación
``` 

---

## 💻 Instalación y Uso

Este proyecto utiliza **uv** para garantizar la instalación reproducible y rápida. 

1. **Clonar y preparar:**

    ```bash 
    git clone https://github.com/Juanpeg1729/genre-classifier.git
    cd genre-classifier
    ```
2. **Instalar dependencias:**
    uv creará automáticamente el entorno virtual y sincronizará las dependencias.

    ```bash
    uv sync
    ```

3. **Ejecutar la aplicación:**
    Para probar el sistema completo necesitarás dos terminales (una para el backend y otra para el frontend).

    **Terminal 1: Leventar la API.** El modelo se descargará automáticamente de Hugging Face la primera vez.

    ```bash
    uv run uvicorn api.main:app --reload
    ```

    La API estará disponible en: http://127.0.0.1:8000/docs

    **Terminal 2: Lanzar el Dashboard.** 

    ```bash
    uv run streamlit run ui/app.py
    ```

    El navegador se abrirá automáticamente en: http://localhost:8501

---

# 🧠 Dashboard & API

El sistema cuenta con dos puntos de entrada:

1. **API REST (FastAPI):** Recibe un JSON con la letra de la canción y devuelve una lista de géneros con sus puntuaciones de confianza. Diseñada para integración M2M (Machine-to-Machine).

2. **Web App (Streamlit):** Una interfaz amigable donde el usuario puede pegar la letra de una canción. Incluye:

    * Validación de entrada.

    * Visualización de resultados con barras de confianza.

    * Feedback visual de carga e inferencia.

---

# ⚙️ Metodología de ML

El núcleo del proyecto es un problema de **Clasificación Multi-Etiqueta** (una canción puede ser Pop y Rock simultáneamente).

1. Ingeniería de Datos:

    * Limpieza de ruido en letras (eliminación de metadatos como [Chorus], [Verse]).

    * Codificación de etiquetas mediante MultiLabelBinarizer (One-Hot Encoding para 80+ géneros).

1. Arquitectura del Modelo:

    * Se utilizó DistilBERT-base-multilingual-cased.

    * Por qué: Ofrece un balance óptimo entre rendimiento (97% de BERT) y velocidad/peso (40% más ligero), crucial para una inferencia en tiempo real.

3. Entrenamiento Optimizado (GPU):

    * Mixed Precision (FP16): Reducción del uso de VRAM a la mitad.

    * Gradient Accumulation: Simulación de batches grandes (Size 16) en hardware limitado.

    * Estrategia de Guardado: Checkpoints automáticos en la nube y recuperación ante fallos.

4. Ajuste de Umbral:

    * Dado que es un problema multi-label, se optimizó el umbral de decisión (Threshold = 0.2) para maximizar el F1-Score, evitando falsos negativos comunes en modelos conservadores.

---

# 📊 Entrenamiento y Resultados

El modelo fue entrenado con un dataset de 5.000 canciones.

* Pérdida (Loss): Se utilizó BCEWithLogitsLoss (Binary Cross Entropy) adaptada para clasificación multi-etiqueta.

* Métricas: Se priorizó el F1-Macro y el ROC-AUC para evaluar el rendimiento en clases desbalanceadas.

* Resultados: El modelo demuestra una capacidad sólida para distinguir géneros principales y subgéneros correlacionados.

El modelo final está alojado públicamente en Hugging Face Hub para facilitar su despliegue sin sobrecargar el repositorio.

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juanpeg1729](https://github.com/Juanpeg1729)
* **LinkedIn:** [Perfil de LinkedIn](https://www.linkedin.com/in/juan-pedro-garcía-sanz-443b31343)
* **Hugging Face:** [@Juanpeg1279](https://huggingface.co/Juanpeg1729)