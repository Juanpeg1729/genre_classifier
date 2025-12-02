# Usamos una imagen ligera de Python 3.11
FROM python:3.11-slim-bookworm

# Instalamos uv directamente desde su imagen oficial (el método más rápido y seguro)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# 1. Copiamos primero los archivos de dependencias (para aprovechar el caché de Docker)
COPY pyproject.toml uv.lock ./

# 2. Instalamos las dependencias del sistema
RUN uv sync --frozen --no-dev

# 3. Copiamos el resto del código
COPY . .

# Añadimos el entorno virtual al PATH para no tener que escribir "uv run" todo el rato
ENV PATH="/app/.venv/bin:$PATH"

# Este comando se sobreescribirá en el docker-compose, pero lo dejamos por defecto
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]