.PHONY: install api ui docker-build docker-up docker-down

install:
	uv sync

api:
	uv run uvicorn api.main:app --reload

ui:
	uv run streamlit run ui/app.py

# --- DOCKER ---
docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down