import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import limpiar_texto

# Modelo de Clasificación de Géneros Musicales desde Hugging Face
MODEL_ID = "Juanpeg1729/genre-classifier"  # ID del modelo en Hugging Face

class GenrePredictor:
    def __init__(self):
        # Asigno el dispositivo (GPU si está disponible, sino CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando modelo desde {MODEL_ID} en {self.device}...")
        
        # Descargo el tokenizador y el modelo (o cargo desde caché si ya está descargado)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) # Cargo el tokenizador
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID) # Cargo el modelo
        self.model.to(self.device) # Muevo el modelo al dispositivo adecuado (GPU/CPU)
        self.model.eval() # Pongo el modelo en modo evaluación (desactiva dropout, etc.)

    def predict(self, text, threshold=0.2):
        clean_text = limpiar_texto(text) # Limpio el texto usando la función definida en preprocessing.py
        
        # Tokenizamos
        inputs = self.tokenizer(
            clean_text, 
            return_tensors="pt", # Usar tensores de PyTorch
            truncation=True, # Truncar si es muy largo
            padding=True, # Padding automático. Esto es útil para batchs. Lo que hace es rellenar con ceros hasta el tamaño del input más largo
            max_length=512 # Longitud máxima del input
        ).to(self.device) # Muevo los tensores al dispositivo adecuado (GPU/CPU)

        # Inferencia (sin calcular gradientes)
        with torch.no_grad():
            logits = self.model(**inputs).logits # **inputs desempaqueta el diccionario de inputs.
        
        # Convertir logits a probabilidades (Sigmoide)
        probs = torch.sigmoid(logits).cpu().numpy()[0] # Muevo a CPU y convierto a numpy array
        
        # Filtrar por umbral y formatear. Este bloque crea una lista de diccionarios con los géneros y sus puntuaciones
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