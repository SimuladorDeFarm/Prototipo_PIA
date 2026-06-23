"""Inferencia: predecir la emoción de un audio con el modelo entrenado.

Reúne las dos piezas ya entrenadas/congeladas: HuBERT (extractor) + la cabeza
clasificadora guardada en el checkpoint. Dado un .wav, extrae su embedding,
lo pasa por la cabeza y devuelve la emoción predicha con su confianza
(probabilidades softmax sobre las 7 clases).

Esto es SOLO inferencia: no entrena ni modifica nada (ver Regla 6 del notebook).
El audio de entrada debe estar en el mismo formato que el de entrenamiento
(mono, 16 kHz); idealmente preprocesado igual (normalizado y con silencio
recortado) para que la predicción sea representativa.
"""

import torch
import torch.nn.functional as F

from . import config
from .clasificador import CabezaEmocion
from .embeddings import cargar_audio, extraer_embedding
from .modelo import cargar_hubert

# Nombres de las emociones en español para mostrar al usuario.
NOMBRES_ES = {
    "neutral": "neutral",
    "joy": "felicidad",
    "sadness": "tristeza",
    "anger": "enojo",
    "fear": "miedo",
    "disgust": "disgusto",
    "surprise": "sorpresa",
}


def cargar_clasificador(ruta_checkpoint=None, dispositivo=None):
    """Carga la cabeza clasificadora desde el checkpoint. Devuelve (modelo, estado)."""
    ruta = ruta_checkpoint or config.CHECKPOINT_FILE
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {ruta}. Entrena el modelo antes (python main.py).")

    dispositivo = dispositivo or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    estado = torch.load(ruta, map_location=dispositivo, weights_only=False)

    modelo = CabezaEmocion(
        dim_entrada=estado["embedding_dim"],
        dim_oculta=estado["hidden_dim"],
        num_clases=len(estado["emociones"]),
        dropout=estado["dropout"],
    ).to(dispositivo)
    modelo.load_state_dict(estado["model_state"])
    modelo.eval()
    return modelo, estado


class Predictor:
    """Carga HuBERT + la cabeza una sola vez y predice sobre uno o varios audios."""

    def __init__(self, ruta_checkpoint=None, device=None):
        self.clasificador, estado = cargar_clasificador(ruta_checkpoint, device)
        self.dispositivo = next(self.clasificador.parameters()).device
        self.emociones = estado["emociones"]
        self.metadata = estado.get("metadata", {})
        # HuBERT congelado, cargado en silencio.
        self.hubert, _ = cargar_hubert(device=str(self.dispositivo), verbose=False)

    @torch.no_grad()
    def predecir(self, ruta_audio):
        """Devuelve un dict con la emoción predicha y las probabilidades por clase."""
        waveform = cargar_audio(ruta_audio)
        embedding = extraer_embedding(self.hubert, self.dispositivo, waveform)

        x = torch.from_numpy(embedding).unsqueeze(0).to(self.dispositivo)
        probs = F.softmax(self.clasificador(x), dim=1).squeeze(0).cpu().numpy()

        ranking = sorted(
            ((self.emociones[i], float(probs[i])) for i in range(len(self.emociones))),
            key=lambda x: x[1], reverse=True,
        )
        emocion, confianza = ranking[0]
        return {
            "emocion": emocion,
            "emocion_es": NOMBRES_ES.get(emocion, emocion),
            "confianza": confianza,
            "ranking": ranking,
        }
