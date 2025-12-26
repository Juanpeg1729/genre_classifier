from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import GenrePredictor

app = FastAPI(title="Spotify Genre Classifier API")

# Cargo el modelo al iniciar
print("Iniciando API y cargando modelo...")
predictor = GenrePredictor()

# Definimos la forma de los datos de entrada
class SongRequest(BaseModel):
    lyrics: str

@app.post("/predict")
def predict_genre(request: SongRequest):
    # Recibe un JSON con la letra de la canción y predice el género
    text = request.lyrics
    results = predictor.predict(text)
    return {"genres": results}

@app.get("/") # Endpoint raíz para verificar que la API está funcionando
def home():
    return {"message": "API de Clasificación de Géneros Musicales funcionando 🚀"}