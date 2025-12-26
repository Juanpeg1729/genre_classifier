import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import limpiar_texto

# Modelo de Clasificación de Géneros Musicales desde Hugging Face
MODEL_ID = "Juanpeg1729/genre-classifier"
LOCAL_MODEL_PATH = Path("./model")

# Token de Hugging Face (necesario si el modelo es privado o para evitar límites)
HF_TOKEN = os.getenv("HF_TOKEN")

class GenrePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Comprobamos si existe 'config.json' dentro de la carpeta local
        if (LOCAL_MODEL_PATH / "config.json").exists():
            print(f"✅ Modelo encontrado en caché local: {LOCAL_MODEL_PATH}")
            model_to_load = LOCAL_MODEL_PATH
        else:
            print(f"⬇️ Modelo no encontrado en local. Descargando de Hugging Face: {MODEL_ID}...")
            model_to_load = MODEL_ID

        # Cargamos el modelo y el tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load, token=HF_TOKEN)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_to_load, token=HF_TOKEN)
        
        # Guardamos en local si acabamos de descargar de internet
        if model_to_load == MODEL_ID:
            print(f"💾 Guardando modelo en {LOCAL_MODEL_PATH} para futuras ejecuciones...")
            # Nos aseguramos de crear la carpeta si no existe
            LOCAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            self.model.save_pretrained(LOCAL_MODEL_PATH)

        # Mover a GPU/CPU y modo evaluación
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, threshold=0.2):
        if not text or not isinstance(text, str):
            return []

        clean_text = limpiar_texto(text)
        
        # Tokenizamos
        inputs = self.tokenizer(
            clean_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)

        # Inferencia
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Post-procesado
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        results = []
        id2label = self.model.config.id2label
        
        for idx, score in enumerate(probs):
            if score > threshold:
                results.append({
                    "label": id2label[idx],
                    "score": float(score)
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)