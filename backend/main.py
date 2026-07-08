import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.src.inferencia import Predictor
from models.verificar_modelos import RUTAS_MODELOS, modelo_disponible, verificar_todos
from preprocessing import preprocesar

# Audio fijo para pruebas: Actor_01, emoción joy (03)
TEST_AUDIO = Path(__file__).parent.parent / "dataset" / "Actor_01" / "03-01-03-01-01-01-01.wav"

_predictor: Predictor | None = None
_predictor_rostro = None  # PredictorRostro | None (import perezoso: py-feat es pesado)
_predictor_texto = None   # PredictorTexto | None (import perezoso: transformers/BETO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _predictor_rostro, _predictor_texto
    print("Verificando modelos...")
    verificar_todos()

    print("Cargando modelos...")

    # Los tres módulos son opcionales: si falta el checkpoint, ese módulo
    # queda deshabilitado (503) pero los demás siguen funcionando.
    if modelo_disponible("voz"):
        _predictor = Predictor(ruta_checkpoint=RUTAS_MODELOS["voz"])
        print("Modelo de voz listo.")
    else:
        _predictor = None
        print(f"[aviso] Módulo de voz no disponible: falta {RUTAS_MODELOS['voz']}")

    try:
        from models.src.inferencia_rostro import PredictorRostro
        if modelo_disponible("rostro"):
            _predictor_rostro = PredictorRostro(ruta_pesos=RUTAS_MODELOS["rostro"])
            print(f"Modelo de rostro listo ({_predictor_rostro.version}).")
        else:
            _predictor_rostro = None
            print(f"[aviso] Módulo de rostro no disponible: falta {RUTAS_MODELOS['rostro']}")
    except Exception as e:
        _predictor_rostro = None
        print(f"[aviso] Módulo de rostro no disponible: {e}")

    try:
        from models.src.inferencia_texto import PredictorTexto
        if modelo_disponible("texto"):
            _predictor_texto = PredictorTexto(ruta_checkpoint=RUTAS_MODELOS["texto"])
            print("Modelo de texto listo.")
        else:
            _predictor_texto = None
            print(f"[aviso] Módulo de texto no disponible: falta {RUTAS_MODELOS['texto']}")
    except Exception as e:
        _predictor_texto = None
        print(f"[aviso] Módulo de texto no disponible: {e}")

    print("Modelos listos.")
    yield
    _predictor = None
    _predictor_rostro = None
    _predictor_texto = None


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {e}")
    try:
        resultado = _inferir(waveform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")
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
            detail="Módulo de rostro no disponible (sin modelo o dependencias faltantes).",
        )
    from models.src.inferencia_rostro import RostroNoValido

    contenido = await imagen.read()
    try:
        resultado = _predictor_rostro.predecir_bytes(contenido)
    except RostroNoValido as e:
        raise HTTPException(status_code=422, detail=str(e))

    resultado["ranking"] = [list(par) for par in resultado["ranking"]]
    return resultado


class TextoInput(BaseModel):
    texto: str


@app.post("/predecir_texto")
async def predecir_texto(payload: TextoInput):
    """Recibe un texto y devuelve la emoción predicha."""
    if _predictor_texto is None:
        raise HTTPException(
            status_code=503,
            detail="Módulo de texto no disponible (dependencias faltantes o sin modelo).",
        )
    if not payload.texto or not payload.texto.strip():
        raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")

    resultado = _predictor_texto.predecir(payload.texto)
    resultado["ranking"] = [list(par) for par in resultado["ranking"]]
    return resultado


@app.get("/")
def home():
    endpoints = ["/predecir (POST)", "/test (GET)"]
    if _predictor_rostro is not None:
        endpoints.append("/predecir_rostro (POST)")
    if _predictor_texto is not None:
        endpoints.append("/predecir_texto (POST)")
    return {"estado": "activo", "endpoints": endpoints}
