# 🎵 Clasificador de Géneros Musicales con IA

![Status](https://img.shields.io/badge/status-in--progress-green)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![uv](https://img.shields.io/badge/uv-enabled-purple)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-RoBERTa-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Este proyecto implementa un sistema completo para clasificar géneros musicales usando Inteligencia Artificial y Procesamiento de Lenguaje Natural. El modelo analiza la letra de una canción y predice automáticamente sus géneros musicales (puede asignar múltiples géneros a una misma canción).

El sistema separa el entrenamiento del modelo (realizado en GPU en la nube) de su uso en producción (desplegado con Docker), permitiendo que el repositorio sea ligero mientras el modelo se descarga automáticamente desde Hugging Face.

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

El proyecto utiliza tecnologías modernas para crear un sistema robusto y escalable:

* **Lenguaje:** Python 3.11
* **Gestión de Dependencias:** [uv](https://github.com/astral-sh/uv) - Gestor de paquetes de alto rendimiento
* **Modelo de IA:** 
    * **Hugging Face Transformers:** Para el fine tuning del modelo RoBERTa
    * **PyTorch:** Framework de deep learning
* **Almacenamiento del modelo:** Hugging Face Hub (el modelo se descarga automáticamente al iniciar)
* **Interfaces:** 
    * **FastAPI:** API REST para predicciones
    * **Streamlit:** Interfaz web interactiva
* **Entrenamiento:** Google Colab con GPU T4
* **Despliegue:** Docker y Docker Compose

---

## 📂 Estructura del Proyecto

El código está organizado en módulos separados para facilitar el mantenimiento:

```text
.
├── api/                 # API REST (FastAPI)
│   ├── __init__.py
│   └── main.py          # Endpoints con validación de datos (Pydantic)
├── model/               # Caché local del modelo (se crea automáticamente)
├── notebooks/           # Notebook con el proceso de entrenamiento
│   └── genre_classification_colab.ipynb   # Pipeline: Carga, Limpieza, Entrenamiento y Subida
├── src/                 # Lógica principal del proyecto
│   ├── __init__.py
│   ├── predictor.py     # Gestión del modelo (descarga y predicciones)
│   └── preprocessing.py # Limpieza de texto
├── ui/                  # Interfaz web (Streamlit)
│   └── app.py           # Aplicación web
├── .env.example         # Plantilla para variables de entorno
├── .dockerignore        # Archivos excluidos de Docker
├── .gitignore           # Archivos excluidos de Git
├── docker-compose.yml   # Configuración de contenedores
├── Dockerfile           # Imagen de Docker
├── Makefile             # Comandos simplificados
├── pyproject.toml       # Dependencias del proyecto
├── uv.lock              # Versiones exactas de dependencias
└── README.md            # Documentación
``` 

---

## 🕹️ Automatización (Makefile)

El proyecto incluye comandos simplificados para facilitar su uso:

| Comando | Descripción |
| :--- | :--- |
| `make help` | Muestra todos los comandos disponibles |
| `make install` | Instala las dependencias del proyecto |
| `make api` | Inicia el servidor API en local |
| `make ui` | Inicia la interfaz web |
| `make docker-build` | Construye las imágenes de Docker |
| `make docker-up` | Inicia todo el sistema con Docker |
| `make docker-down` | Detiene todos los contenedores |
| `make clean` | Limpia archivos temporales y caché |

---

## 💻 Instalación y Uso

### Configuración inicial (opcional)

Si quieres usar tu token de Hugging Face (recomendado para evitar límites de descarga):

1. Copia el archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```

2. Edita `.env` y añade tu token. Puedes obtenerlo en https://huggingface.co/settings/tokens
   ```
   HF_TOKEN=hf_tu_token_aqui
   ```

**Nota:** El token es opcional para modelos públicos, pero ayuda a evitar límites de descarga.

---

### Opción A: Docker (Recomendada)

1. Inicia el sistema completo:

    ```bash
    make docker-up
    ```

    La primera vez descargará las imágenes y el modelo (~500MB). Puede tardar unos minutos.

2. **Acceder a las interfaces:**

    * Interfaz web: http://localhost:8501
    * API: http://localhost:8000/docs

3. **Detener el sistema:**

    ```bash
    make docker-down
    ```

### Opción B: Ejecución Local

Para desarrollo o si prefieres ejecutar sin Docker:

1. **Instalar dependencias:**

    ```bash
    make install
    ```

2. **Ejecutar servicios (en terminales separadas):**

    ```bash
    make api  # Terminal 1: Inicia la API
    make ui   # Terminal 2: Inicia la interfaz web
    ```

3. **Acceder a las interfaces:**

    * Interfaz web: http://localhost:8501
    * API: http://localhost:8000/docs

**Nota:** El modelo se descarga automáticamente la primera vez y se guarda en `model/` para ejecuciones futuras.

---

## 🧠 Dashboard & API

El sistema ofrece dos formas de interactuar con el modelo:

### 1. Dashboard Interactivo (Streamlit)

Interfaz web simple y visual:

* **Entrada de texto:** Área para pegar la letra de la canción
* **Visualización:** Barras de progreso que muestran la probabilidad de cada género
* **Tiempo real:** Indicadores de carga durante el análisis

### 2. API REST (FastAPI)

Endpoint programático para integraciones:

* **Endpoint `/predict`:** Recibe la letra en formato JSON y devuelve los géneros detectados
* **Validación automática:** Verifica que los datos de entrada sean correctos
* **Documentación interactiva:** Interfaz Swagger en `/docs` para probar la API directamente desde el navegador

---

## ⚙️ Metodología de Data Science

El principal desafío de este proyecto fue la calidad de los datos. Se aplicó un enfoque centrado en datos para mejorar significativamente el rendimiento del modelo.

### 1. Ingeniería de Datos y Limpieza:

* **Filtrado de idioma:** El dataset contenía múltiples idiomas. Se filtró para conservar solo canciones en inglés (97% del total), optimizando el uso del modelo RoBERTa.

* **Agrupación de géneros:** El dataset original tenía 88 subgéneros desbalanceados (ej: cloud rap, trap, drill). Se consolidaron en 14 géneros principales (Hip-Hop, Rock, Pop, Metal, etc.), mejorando la capacidad de aprendizaje del modelo.

* **Limpieza de texto:** Se eliminaron metadatos de las letras (ej: [Chorus], [Verse 1]) usando expresiones regulares.

### 2. Modelado:

* **Arquitectura:** RoBERTa (versión optimizada de BERT). Se eligió por su capacidad superior para entender contextos complejos, ironía y slang en inglés.

* **Clasificación multi-etiqueta:** El modelo puede asignar múltiples géneros a una misma canción (ej: Rock + Alternative).

---

## 📊 Entrenamiento y Resultados

El modelo fue entrenado en Google Colab usando GPU T4:

* **Dataset:** 50,000 canciones aproximadamente

* **Optimizaciones:** Precisión mixta (FP16) y acumulación de gradientes

* **Métricas:**
    * **Umbral optimizado:** 0.2 para maximizar la precisión en clasificación multi-etiqueta
    * **ROC-AUC:** Superior a 0.90, indicando excelente capacidad de clasificación

El modelo entrenado está disponible públicamente en: [Juanpeg1729/genre-classifier](https://huggingface.co/Juanpeg1729/genre-classifier)

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juanpeg1729](https://github.com/Juanpeg1729)
* **LinkedIn:** [Perfil de LinkedIn](https://www.linkedin.com/in/juanpedrogarciasanz)
* **Hugging Face:** [@Juanpeg1279](https://huggingface.co/Juanpeg1729)