import re

def limpiar_texto(texto):
    # Función para limpiar el texto de letras de canciones. Elimina corchetes y su contenido.
    return re.sub(r"\[.*?\]", "", texto).strip()