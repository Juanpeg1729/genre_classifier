.PHONY: help install api ui docker-build docker-up docker-down clean

.DEFAULT_GOAL := help

help: ## Muestra esta ayuda
	@echo "Comandos disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Instala las dependencias del proyecto
	uv sync

api: ## Inicia el servidor API con uvicorn
	uv run uvicorn api.main:app --reload

ui: ## Inicia la interfaz de usuario con streamlit
	uv run streamlit run ui/app.py

docker-build: ## Construye las imágenes de Docker
	docker-compose build

docker-up: ## Levanta los contenedores de Docker
	docker-compose up

docker-down: ## Detiene los contenedores de Docker
	docker-compose down

clean: ## Limpia archivos de caché de Python
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Caché limpiado"