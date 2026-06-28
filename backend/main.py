import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models.src.inferencia import Predictor
from preprocessing import preprocesar

CHECKPOINT = Path(__file__).parent / "models" / "clasificador_voz.pt"
CHECKPOINT_ROSTRO = Path(__file__).parent / "models" / "clasificador_rostro.joblib"

# Audio fijo para pruebas: Actor_01, emoción joy (03)
TEST_AUDIO = Path(__file__).parent.parent / "dataset" / "Actor_01" / "03-01-03-01-01-01-01.wav"

_predictor: Predictor | None = None
_predictor_rostro = None  # PredictorRostro | None (import perezoso: py-feat es pesado)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _predictor_rostro
    print("Cargando modelos...")
    _predictor = Predictor(ruta_checkpoint=CHECKPOINT)

    # El módulo de rostro es opcional: si py-feat no está instalado o no hay
    # modelo entrenado, el backend de voz sigue funcionando igualmente.
    try:
        from models.src.inferencia_rostro import PredictorRostro
        _predictor_rostro = PredictorRostro(ruta_bundle=CHECKPOINT_ROSTRO)
        print(f"Modelo de rostro listo (bundle {_predictor_rostro.version}).")
    except Exception as e:
        _predictor_rostro = None
        print(f"[aviso] Módulo de rostro no disponible: {e}")

    print("Modelos listos.")
    yield
    _predictor = None
    _predictor_rostro = None


app = FastAPI(title="PIA — Predicción de Emociones", lifespan=lifespan)

# CORS abierto: prototipo preliminar, el frontend corre en otro origen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/predecir_rostro")
async def predecir_rostro(imagen: UploadFile = File(...)):
    """Recibe una imagen con un rostro y devuelve la emoción predicha."""
    if _predictor_rostro is None:
        raise HTTPException(
            status_code=503,
            detail="Módulo de rostro no disponible (py-feat no instalado o sin modelo).",
        )
    from models.src.inferencia_rostro import RostroNoValido

    contenido = await imagen.read()
    sufijo = Path(imagen.filename or "").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=sufijo, delete=False)
    try:
        tmp.write(contenido)
        tmp.close()
        resultado = _predictor_rostro.predecir(tmp.name)
    except RostroNoValido as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    resultado["ranking"] = [list(par) for par in resultado["ranking"]]
    return resultado


@app.get("/")
def home():
    endpoints = ["/predecir (POST)", "/test (GET)"]
    if _predictor_rostro is not None:
        endpoints.append("/predecir_rostro (POST)")
    return {"estado": "activo", "endpoints": endpoints}
