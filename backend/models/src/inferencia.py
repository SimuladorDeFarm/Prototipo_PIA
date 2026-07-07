"""Inferencia del módulo de voz (HuBERT fine-tuned v4) para el prototipo web.

Pipeline: audio bytes → preprocesar (mono, 16kHz, RMS norm, trim) → pad/truncar
→ HuBERTEmotionModel → 7 probabilidades → emoción

Replica la arquitectura de entrenar_v4.py (PIA_modulo_voz):
HuBERT con 4 capas descongeladas + cabeza (768→512→128→7).
El state_dict incluye HuBERT + cabeza juntos.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
EMBEDDING_DIM = 768
EMOCIONES_7 = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
MAX_LEN_SAMPLES = 48000  # 3 segundos a 16kHz

NOMBRES_ES = {
    "neutral": "neutral",
    "joy": "felicidad",
    "sadness": "tristeza",
    "anger": "enojo",
    "fear": "miedo",
    "disgust": "disgusto",
    "surprise": "sorpresa",
}


class HuBERTEmotionModel(nn.Module):
    """HuBERT + cabeza clasificadora, con fine-tuning parcial."""

    def __init__(self, num_clases=7, unfreeze_layers=4):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(HUBERT_MODEL_NAME)

        for param in self.hubert.parameters():
            param.requires_grad = False

        n_layers = len(self.hubert.encoder.layers)
        for i in range(n_layers - unfreeze_layers, n_layers):
            for param in self.hubert.encoder.layers[i].parameters():
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(EMBEDDING_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_clases),
        )

    def forward(self, waveforms):
        outputs = self.hubert(waveforms)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.head(embeddings)


class Predictor:
    """Carga HuBERTEmotionModel v4 una vez y predice sobre audios."""

    def __init__(self, ruta_checkpoint=None, device=None):
        self.dispositivo = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if ruta_checkpoint is None or not ruta_checkpoint.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint de voz: {ruta_checkpoint}"
            )

        self.modelo = HuBERTEmotionModel(
            num_clases=len(EMOCIONES_7), unfreeze_layers=4
        ).to(self.dispositivo)

        state = torch.load(ruta_checkpoint, map_location=self.dispositivo, weights_only=False)
        self.modelo.load_state_dict(state)
        self.modelo.eval()

    @torch.no_grad()
    def predecir(self, ruta_audio):
        """Recibe ruta a WAV preprocesado y devuelve predicción."""
        import soundfile as sf
        wav, sr = sf.read(str(ruta_audio), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype(np.float32)

        if len(wav) > MAX_LEN_SAMPLES:
            wav = wav[:MAX_LEN_SAMPLES]
        elif len(wav) < MAX_LEN_SAMPLES:
            wav = np.pad(wav, (0, MAX_LEN_SAMPLES - len(wav)), mode="constant")

        tensor = torch.from_numpy(wav).unsqueeze(0).to(self.dispositivo)
        logits = self.modelo(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        ranking = sorted(
            ((EMOCIONES_7[i], float(probs[i])) for i in range(len(EMOCIONES_7))),
            key=lambda x: x[1],
            reverse=True,
        )
        emocion, confianza = ranking[0]

        return {
            "emocion": emocion,
            "emocion_es": NOMBRES_ES.get(emocion, emocion),
            "confianza": confianza,
            "ranking": ranking,
        }
