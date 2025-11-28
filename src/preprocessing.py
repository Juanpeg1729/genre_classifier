import re

def limpiar_texto(texto):
    # La misma función exacta que usaste en el entrenamiento
    return re.sub(r"\[.*?\]", "", texto).strip()