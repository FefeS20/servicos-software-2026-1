import os
import shutil
import whisper
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Carrega o modelo (o 'base' ou 'tiny' é rápido de baixar e ideal para uma aula de 4h)
print("Carregando modelo de IA (Whisper)...")
model = whisper.load_model("base")
print("Modelo carregado!")


@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    # 1. Salva o arquivo temporariamente no container
    caminho_temp = f"temp_{file.filename}"
    with open(caminho_temp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. IA converte o áudio para texto.
        # Forçamos o idioma para português para maior precisão e rapidez.
        resultado = model.transcribe(caminho_temp, language="pt")
        texto = resultado["text"].strip()
    finally:
        # 3. Limpa o arquivo de áudio temporário para não lotar o container
        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)

    # Retorna o texto que servirá como prompt para o próximo passo do algoritmo
    return {"texto": texto}
