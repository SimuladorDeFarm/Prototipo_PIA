"""Inferencia del módulo de texto (BETO fine-tuned v4) para el prototipo web.

Pipeline: texto crudo → convertir emojis a español → eliminar URLs/placeholders
→ tokenizar con BETO (max_length=128, padding="max_length") → BETO → 7 probabilidades → emoción

Replica el pipeline descrito en Docs/pipeline_texto.md:
BETO (dccuchile/bert-base-spanish-wwm-cased) full fine-tune + Focal Loss.
"""

import re
from pathlib import Path

import emoji
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128

LABEL2ID = {"others": 0, "joy": 1, "sadness": 2, "anger": 3, "fear": 4, "disgust": 5, "surprise": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
EMOCIONES = [ID2LABEL[i] for i in range(len(ID2LABEL))]

# "others" es la clase neutral de EMOEvent; se traduce para mostrarla en español.
EMOCIONES_ES = {
    "others": "neutral",
    "joy": "felicidad",
    "sadness": "tristeza",
    "anger": "enojo",
    "fear": "miedo",
    "disgust": "disgusto",
    "surprise": "sorpresa",
}

_DELIM = ("\x00", "\x01")
_TOKEN_EMOJI = re.compile("\x00([^\x01]*)\x01")
_PATRON_PLACEHOLDERS = re.compile(r"\b(?:HASHTAG|URL|USER)\b")
_PATRON_URL = re.compile(r"http\S+")


def _convertir_emojis(texto: str) -> str:
    demoj = emoji.demojize(texto, language="es", delimiters=_DELIM)
    convertido = _TOKEN_EMOJI.sub(
        lambda m: " " + m.group(1).replace("_", " ").strip() + " ", demoj
    )
    return emoji.replace_emoji(convertido, "")


def _eliminar_placeholders(texto: str) -> str:
    sin = _PATRON_URL.sub(" ", texto)
    sin = _PATRON_PLACEHOLDERS.sub(" ", sin)
    sin = sin.replace("#", "")
    return re.sub(r"\s{2,}", " ", sin).strip()


def _preprocesar(texto: str) -> str:
    # El orden importa: emojis antes que URLs (ver Docs/pipeline_texto.md).
    texto = _convertir_emojis(texto)
    return _eliminar_placeholders(texto)


class PredictorTexto:
    """Carga BETO v4 una vez y predice sobre texto en español."""

    def __init__(self, ruta_checkpoint=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ruta = Path(ruta_checkpoint) if ruta_checkpoint else None
        if ruta is None or not ruta.exists():
            raise FileNotFoundError(f"No se encontró el checkpoint de texto: {ruta}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        state = torch.load(ruta, map_location=self.device, weights_only=True)
        self.modelo.load_state_dict(state)
        self.modelo.to(self.device)
        self.modelo.eval()

    @torch.no_grad()
    def predecir(self, texto: str) -> dict:
        """Recibe texto crudo y devuelve la predicción."""
        texto_limpio = _preprocesar(texto)
        inputs = self.tokenizer(
            texto_limpio,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.modelo(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        ranking = sorted(
            ((EMOCIONES[i], float(probs[i])) for i in range(len(EMOCIONES))),
            key=lambda x: x[1],
            reverse=True,
        )
        emocion, confianza = ranking[0]

        return {
            "emocion": emocion,
            "emocion_es": EMOCIONES_ES[emocion],
            "confianza": confianza,
            "ranking": ranking,
        }
