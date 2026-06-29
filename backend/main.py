from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, BertForSequenceClassification

import re
import emoji

# --- Configuración ---------------------------------------------------------
CHECKPOINT = Path(__file__).parent / "models" / "beto_emoevent_best.pth"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128

ID2LABEL = {0:"others", 1:"joy", 2:"sadness", 3:"anger", 4:"fear", 5:"disgust", 6:"surprise"}
NOMBRES_ES = {"others":"neutral", "joy":"felicidad", "sadness":"tristeza",
              "anger":"enojo", "fear":"miedo", "disgust":"disgusto", "surprise":"sorpresa"}

# --- Preprocesamiento ------------------------------------------------------
_DELIM = ("\x00", "\x01")
_TOKEN_EMOJI = re.compile("\x00([^\x01]*)\x01")
_PATRON_URL  = re.compile(r"http\S+")
_PATRON_PLAC = re.compile(r"\b(?:HASHTAG|URL|USER)\b")

def preprocesar(texto: str) -> str:
    demoj = emoji.demojize(texto, language="es", delimiters=_DELIM)
    texto = _TOKEN_EMOJI.sub(lambda m: " " + m.group(1).replace("_", " ").strip() + " ", demoj)
    texto = emoji.replace_emoji(texto, "")
    texto = _PATRON_URL.sub(" ", texto)
    texto = _PATRON_PLAC.sub(" ", texto)
    texto = texto.replace("#", "")
    return re.sub(r"\s{2,}", " ", texto).strip()

# --- Estado global ---------------------------------------------------------
_tokenizer = None
_modelo    = None
_device    = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tokenizer, _modelo, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cargando tokenizador...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Cargando modelo...")
    checkpoint = torch.load(CHECKPOINT, map_location=_device, weights_only=False)
    label2id = checkpoint.get("label2id", {v: k for k, v in ID2LABEL.items()})
    id2label  = {v: k for k, v in label2id.items()}
    _modelo = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=7, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    _modelo.load_state_dict(checkpoint["model_state_dict"])
    _modelo.to(_device)
    _modelo.eval()
    print("Modelo listo.")
    yield
    _modelo = None
    _tokenizer = None

# --- App -------------------------------------------------------------------
app = FastAPI(title="PIA — Predicción de Emociones por Texto", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextoInput(BaseModel):
    texto: str

@app.post("/predecir")
async def predecir(payload: TextoInput):
    if not payload.texto or not payload.texto.strip():
        raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")
    
    texto_limpio = preprocesar(payload.texto)
    inputs = _tokenizer(
        texto_limpio,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _modelo(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()
    ranking = sorted(
        [(ID2LABEL[i], float(probs[i])) for i in range(7)],
        key=lambda x: x[1], reverse=True
    )
    emocion, confianza = ranking[0]
    return {
        "emocion": emocion,
        "emocion_es": NOMBRES_ES.get(emocion, emocion),
        "confianza": confianza,
        "ranking": [list(p) for p in ranking],
    }

@app.get("/")
def home():
    return {"estado": "activo", "endpoints": ["/predecir (POST)"], "modulo": "texto"}