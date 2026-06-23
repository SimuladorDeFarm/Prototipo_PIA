import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile

from models.src.inferencia import Predictor
from preprocessing import preprocesar

CHECKPOINT = Path(__file__).parent / "models" / "clasificador_voz.pt"

# Audio fijo para pruebas: Actor_01, emoción joy (03)
TEST_AUDIO = Path(__file__).parent.parent / "dataset" / "Actor_01" / "03-01-03-01-01-01-01.wav"

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    print("Cargando modelos...")
    _predictor = Predictor(ruta_checkpoint=CHECKPOINT)
    print("Modelos listos.")
    yield
    _predictor = None


app = FastAPI(title="PIA — Predicción de Emociones", lifespan=lifespan)


def _inferir(waveform):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, waveform, 16_000)
        tmp.close()
        return _predictor.predecir(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/predecir")
async def predecir(audio: UploadFile = File(...)):
    """Recibe un archivo de audio y devuelve la emoción predicha."""
    contenido = await audio.read()
    try:
        waveform = preprocesar(contenido)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    resultado = _inferir(waveform)
    resultado["ranking"] = [list(par) for par in resultado["ranking"]]
    return resultado


@app.get("/test")
async def test():
    """Corre inferencia sobre el audio de prueba fijo (Actor_01, joy)."""
    if not TEST_AUDIO.exists():
        raise HTTPException(status_code=404, detail=f"Audio de prueba no encontrado: {TEST_AUDIO}")
    with open(TEST_AUDIO, "rb") as f:
        contenido = f.read()
    waveform = preprocesar(contenido)
    resultado = _inferir(waveform)
    resultado["ranking"] = [list(par) for par in resultado["ranking"]]
    resultado["archivo_prueba"] = TEST_AUDIO.name
    return resultado


@app.get("/")
def home():
    return {"estado": "activo", "endpoints": ["/predecir (POST)", "/test (GET)"]}
