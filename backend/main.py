import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models.src.inferencia import Predictor
from preprocessing import preprocesar

# Rutas configuradas de forma relativa para evitar errores
BASE_DIR = Path(__file__).parent
CHECKPOINT = BASE_DIR / "models" / "clasificador_voz.pt"
MODELO_ROSTRO = joblib.load(BASE_DIR / "models" / "random_forest.joblib")

TEST_AUDIO = BASE_DIR.parent / "dataset" / "Actor_01" / "03-01-03-01-01-01-01.wav"

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
    contenido = await audio.read()
    try:
        waveform = preprocesar(contenido)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    resultado = _inferir(waveform)
    resultado["ranking"] = [list(par) for par in resultado["ranking"]]
    return resultado

@app.post("/predecir/rostro")
async def predecir_rostro(archivo: UploadFile = File(...)):
    """Recibe una imagen y devuelve la emoción detectada."""
    # Guardamos temporalmente
    with open("temp_image.jpg", "wb") as buffer:
        buffer.write(await archivo.read())
    
    # Aquí iría tu lógica de extracción de rasgos
    return {"mensaje": "Endpoint de rostro configurado", "status": "recibido"}

@app.get("/test")
async def test():
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
    return {"estado": "activo", "endpoints": ["/predecir (POST)", "/test (GET)", "/predecir/rostro (POST)"]}