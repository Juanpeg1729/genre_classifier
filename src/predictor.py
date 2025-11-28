import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import limpiar_texto

# --- CONFIGURACIÓN ---
MODEL_ID = "Juanpeg1729/genre-classifier"  # <--- ¡PON AQUÍ TU REPO_ID EXACTO!

class GenrePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando modelo desde {MODEL_ID} en {self.device}...")
        
        # Esto descarga el modelo de la nube (o usa el caché si ya lo bajaste)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, threshold=0.2):
        clean_text = limpiar_texto(text)
        
        # Tokenizar
        inputs = self.tokenizer(
            clean_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)

        # Inferencia (sin calcular gradientes)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Convertir logits a probabilidades (Sigmoide)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Filtrar por umbral y formatear
        results = []
        id2label = self.model.config.id2label
        
        for idx, score in enumerate(probs):
            if score > threshold:
                results.append({
                    "label": id2label[idx],
                    "score": float(score)
                })
        
        # Ordenar de mayor a menor confianza
        return sorted(results, key=lambda x: x["score"], reverse=True)