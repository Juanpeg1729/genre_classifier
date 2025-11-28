from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import GenrePredictor

app = FastAPI(title="Spotify Genre Classifier API")

# Cargamos el modelo UNA SOLA VEZ al inicio (esto tarda unos segundos)
print("Iniciando API y cargando modelo...")
predictor = GenrePredictor()

class LyricRequest(BaseModel):
    lyrics: str

@app.post("/predict")
def predict_genre(request: LyricRequest):
    """Recibe una letra y devuelve los géneros probables."""
    results = predictor.predict(request.lyrics)
    return {"genres": results}

@app.get("/")
def home():
    return {"message": "API de Clasificación de Géneros Musicales funcionando 🚀"}