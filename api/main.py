from fastapi import FastAPI
from src.predictor import GenrePredictor

app = FastAPI(title="Spotify Genre Classifier API")

# Cargo el modelo al iniciar
print("Iniciando API y cargando modelo...")
predictor = GenrePredictor()

@app.post("/predict") # Este decorador define el endpoint /predict
def predict_genre(lyrics: dict):
    # Recibe un JSON con la letra de la canción y predice el género
    text = lyrics.get("lyrics", "")
    results = predictor.predict(text)
    return {"genres": results}

@app.get("/") # Endpoint raíz para verificar que la API está funcionando
def home():
    return {"message": "API de Clasificación de Géneros Musicales funcionando 🚀"}