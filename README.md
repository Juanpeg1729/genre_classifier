# 🎵 Spotify Genre Classifier: End-to-End NLP Pipeline

![Status](https://img.shields.io/badge/status-in--progress-green)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![uv](https://img.shields.io/badge/uv-enabled-purple)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-RoBERTa-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Este proyecto implementa un sistema End-to-End MLOps para la clasificación de géneros musicales utilizando Procesamiento de Lenguaje Natural (NPL). El modelo es capaz de predecir múltiples géneros (Multi-Label) para una canción basándose únicamente en su letra, utilizando modelos Transformer basado en BERT (RoBERTa).

El repositorio demuestra una arquitectura de software moderna, desacoplando el entrenamiento (realizado en GPU en la nube) de la inferencia (desplegada mediante microservicios Dockerizados), con un enfoque riguroso en la limpieza de datos y la optimización de recursos.

---

## 📋 Tabla de Contenidos

- [Arquitectura y Tech Stack](#-arquitectura-y-tech-stack)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Automatización (Makefile)](#%EF%B8%8F-automatización-makefile)
- [Instalación y Uso (Docker & Local)](#-instalación-y-uso)
- [Dashboard & API](#-dashboard--api)
- [Metodología de Data Science](#-metodología-de-data-science)
- [Entrenamiento y Resultados](#-entrenamiento-y-resultados)
- [Autor](#-autor)
---

## 🛠 Arquitectura y Tech Stack

El proyecto integra herramientas modernas para crear un sistema robusto, modular y escalable:

* **Lenguaje:** Python 3.11
* **Gestión de Dependencias:** [uv](https://github.com/astral-sh/uv) (Gestor de paquetes de alto rendimiento).
* **Modelado (NPL):** 
    * **Hugging Face Transformers:** Fine-tuning de roberta-base.
    * **PyTorch:** motor de cálculo tensorial.
* **Infraestructura del modelo: Hugging Face Hub** (Registro de modelos en la nube). El contenedor descarga el modelo automáticamente al arrancar, manteniendo el repositorio ligero. 
* **Interfaces:** 
    * **FastAPI:** Backend para servir predicciones.
    * **Streamlit:** Frontend interactivo para el usuario final.
* **Entrenamiento:** Google Colab (T4 GPU) con estrategias de ahorro de memoria.
* **Despliegue:** Docker & Docker Compose para orquestación de contenedores.

---

## 📂 Estructura del Proyecto

El código sigue una arquitectura de paquete modular, separando configuración, lógica y presentación:

```text
.
├── api/                 # 🔌 Microservicio Backend (FastAPI)
│   ├── __init__.py
│   └── main.py          # Endpoints con validación Pydantic
├── model/               # 🤖 Caché local del modelo (se crea automáticamente)
├── notebooks/           # 📓 Documentación ejecutable (Training Log)
│   └── training.ipynb   # Pipeline completo: Carga, Limpieza, Training, Upload
├── src/                 # 🧠 Lógica del Negocio compartida
│   ├── __init__.py
│   ├── predictor.py     # Clase que gestiona el modelo (descarga y caché)
│   └── preprocessing.py # Normalización de texto (Regex)
├── ui/                  # 🎨 Microservicio Frontend (Streamlit)
│   └── app.py           # Interfaz de usuario
├── .env.example         # Plantilla para variables de entorno
├── .dockerignore        # Exclusiones para optimizar imágenes
├── .gitignore           # Exclusiones de git
├── docker-compose.yml   # Orquestación de servicios (API + UI)
├── Dockerfile           # Receta de imagen (Multi-stage build con uv)
├── Makefile             # 🕹️ Automatización de comandos
├── pyproject.toml       # Definición de dependencias
├── uv.lock              # Lockfile para reproducibilidad exacta
└── README.md            # Documentación
``` 

---

## 🕹️ Automatización (Makefile)

Para facilitar el uso, el proyecto incluye un Makefile que abstrae los comandos complejos.

| Comando | Descripción |
| :--- | :--- |
| `make help` | Muestra todos los comandos disponibles. |
| `make install` | Instala las dependencias con `uv`. |
| `make api` | Levanta el servidor de la API (FastAPI) en local. |
| `make ui` | Lanza la aplicación web (Streamlit). |
| `make docker-build` | Construye la imagen de Docker. |
| `make docker-up` | Levanta todo el sistema (API + Dashboard) en contenedores. |
| `make docker-down` | Apaga todos los contenedores. |
| `make clean` | Limpia archivos de caché de Python. |

---

## 💻 Instalación y Uso

### Configuración inicial (opcional)

Si quieres usar tu token de Hugging Face (recomendado para evitar límites de descarga):

1. Copia el archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```

2. Edita `.env` y añade tu token (consíguelo en https://huggingface.co/settings/tokens):
   ```
   HF_TOKEN=hf_tu_token_real_aqui
   ```

**Nota:** El token no es necesario si el modelo es público, pero ayuda a evitar límites de descarga.

---

### Opción A: Docker (Recomendada 🐳)

1. Levanta todo el sistema sin preocuparte por dependencias de Python o versiones de CUDA.

    ```bash
    make docker-up
    ```

    (La primera vez tardará unos minutos mientras descarga las imágenes y el modelo RoBERTa de 500MB).

2. **Acceder:**

    * 🎨 Web App: Abre http://localhost:8501 en tu navegador.

    * ⚙️ API Docs: Abre http://localhost:8000/docs.

3. Detener:

    ```bash
    make docker-down
    ```

### Opción B: Ejecución Local (con uv)

Si deseas editar el código o desarrollar localmente:

1. **Instalar dependencias:**

    ```bash
    make install
    ```

2. **Ejecutar servicios (en terminales separadas):**

    ```bash
    make api  # Terminal 1: Inicia la API
    make ui   # Terminal 2: Inicia la interfaz
    ```

3. **Acceder:**

    * 🎨 Web App: http://localhost:8501
    * ⚙️ API Docs: http://localhost:8000/docs

**Nota:** El modelo se descarga automáticamente la primera vez y se guarda en la carpeta `model/` para futuras ejecuciones.

---

## 🧠 Dashboard & API

El sistema expone dos interfaces principales para interactuar con el modelo:

1. **Dashboard Interactivo (Streamlit)**

    Diseñado para usuarios finales. Permite:

    * Entrada de Texto: Un área de texto simple para pegar la letra de la canción a analizar.

    * Visualización de Confianza: Muestra los géneros detectados con barras de progreso que indican la probabilidad (confianza) del modelo para cada etiqueta.

    * Feedback en Tiempo Real: Indicadores de carga mientras el modelo realiza la inferencia.

2. **API REST (FastAPI)**

    El motor del sistema, diseñado para integraciones.

    * Endpoint `/predict`: Acepta un JSON con la letra y devuelve los géneros detectados.

    * Validación automática con Pydantic: Garantiza que los datos de entrada sean correctos.

    * Documentación interactiva: Swagger UI disponible en `/docs` para probar la API desde el navegador.

---

## ⚙️ Metodología de Data Science

El mayor reto de este proyecto no fue el modelo, sino los datos. Se aplicó una estrategia de Data-Centric AI para pasar de un rendimiento pobre a un modelo robusto.

1. **Ingeniería de Datos y Limpieza:**

    * Filtrado de Idioma: Se detectó que el dataset contenía múltiples idiomas. Se utilizó langdetect para filtrar y conservar solo el corpus en inglés (97% del total), optimizando el uso de roberta-base (monolingüe).

    * Agrupación de Géneros (Label Engineering): El dataset original contenía 88 micro-géneros desbalanceados (ej: cloud rap, trap, gangster rap). Se desarrolló un algoritmo de mapeo para consolidarlos en 14 Macro-Géneros sólidos (Hip-Hop, Rock, Pop, Metal, etc.), mejorando drásticamente la señal de aprendizaje.

    * Limpieza de Texto: Eliminación de metadatos de Genius (ej: [Chorus], [Verse 1]) mediante Regex.

2. **Modelado:**

    * Arquitectura: RoBERTa (Robustly optimized BERT approach). Se eligió sobre DistilBERT por su capacidad superior para entender contextos complejos, ironía y slang en inglés.

    * Estrategia Multi-Label: Se utilizó BCEWithLogitsLoss para permitir que una canción pertenezca a múltiples géneros simultáneamente (ej: Rock y Pop).

---

## 📊 Entrenamiento y Resultados

El modelo fue entrenado utilizando Google Colab (T4 GPU) con técnicas de optimización de memoria:

* Dataset: ~50.000 canciones.

* Optimizaciones: Mixed Precision (FP16) y Gradient Accumulation (Batch Size efectivo = 16).

* Métricas:

    * El modelo final utiliza un umbral de decisión optimizado de 0.2. Esto corrige el sesgo conservador de la red neuronal, maximizando el F1-Score y la Accuracy en un problema de clasificación multietiqueta.

    * El modelo final alcanza un ROC-AUC > 0.90, demostrando una excelente capacidad de separación entre clases.

El modelo entrenado se encuentra alojado públicamente en Hugging Face Hub: Juanpeg1729/genre-classifier.

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juanpeg1729](https://github.com/Juanpeg1729)
* **LinkedIn:** [Perfil de LinkedIn](https://www.linkedin.com/in/juanpedrogarciasanz)
* **Hugging Face:** [@Juanpeg1279](https://huggingface.co/Juanpeg1729)