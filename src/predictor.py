import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import limpiar_texto

# Modelo de Clasificación de Géneros Musicales desde Hugging Face
MODEL_ID = "Juanpeg1729/genre-classifier"  # ID del modelo en Hugging Face
LOCAL_MODEL_PATH = "./model"

# Token de Hugging Face (opcional para modelos públicos, necesario para privados)
HF_TOKEN = os.getenv("HF_TOKEN", None)

class GenrePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Comprobamos si existe el archivo 'config.json' en la carpeta local.
        # Si existe, significa que ya lo descargamos alguna vez.
        if os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
            print(f"✅ Modelo encontrado en caché local: {LOCAL_MODEL_PATH}")
            model_to_load = LOCAL_MODEL_PATH
        else:
            print(f"⬇️ Modelo no encontrado en local. Descargando de Hugging Face: {MODEL_ID}...")
            model_to_load = MODEL_ID

        # Cargamos (ya sea del disco o de internet)
        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_to_load)
        
        # Guardamos (Solo si hemos tenido que descargar de internet)
        if model_to_load == MODEL_ID:
            print(f"💾 Guardando modelo en {LOCAL_MODEL_PATH} para futuras ejecuciones...")
            self.tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            self.model.save_pretrained(LOCAL_MODEL_PATH)

        # Configuración final
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, threshold=0.2):
        clean_text = limpiar_texto(text)
        inputs = self.tokenizer(
            clean_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
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