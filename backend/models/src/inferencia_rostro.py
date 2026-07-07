"""Inferencia del módulo de rostro (EfficientNet-B0) para el prototipo web.

Pipeline: imagen bytes → decodificar → detectar cara (YuNet) → recortar
→ resize 224×224 → normalizar → EfficientNet-B0 → 7 probabilidades → emoción

Replica la misma arquitectura de entrenar_v2.py (PIA_modulo_rostro_v2):
EfficientNet-B0 con cabeza personalizada (1280→512→128→7).
"""

import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

EMOCIONES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOCIONES_ES = {
    "anger": "enojo",
    "disgust": "disgusto",
    "fear": "miedo",
    "happy": "alegría",
    "neutral": "neutral",
    "sad": "tristeza",
    "surprise": "sorpresa",
}

MODELS_DIR = Path(__file__).resolve().parent.parent
YUNET_PATH = MODELS_DIR / "rostro" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class RostroNoValido(ValueError):
    """No se detectó un rostro válido en la imagen."""


def _crear_efficientnet(num_clases=7):
    modelo = models.efficientnet_b0(weights=None)
    in_features = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_clases),
    )
    return modelo


def _descargar_yunet():
    if YUNET_PATH.exists():
        return
    import urllib.request
    YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Descargando YuNet (~230KB)...")
    urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
    print(f"  YuNet descargado en {YUNET_PATH}")


class PredictorRostro:
    """Carga EfficientNet-B0 + YuNet una vez y predice sobre imágenes."""

    def __init__(self, ruta_pesos=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = "rostro-efficientnet-v2"

        ruta = Path(ruta_pesos) if ruta_pesos else (MODELS_DIR / "rostro" / "mejor_modelo_v2.pt")
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {ruta}")

        self.modelo = _crear_efficientnet()
        self.modelo.load_state_dict(torch.load(ruta, map_location=self.device, weights_only=True))
        self.modelo.to(self.device)
        self.modelo.eval()

        _descargar_yunet()

    def _detectar_caras(self, imagen_bgr):
        alto, ancho = imagen_bgr.shape[:2]
        detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (ancho, alto))
        detector.setScoreThreshold(0.5)
        _, detecciones = detector.detect(imagen_bgr)
        if detecciones is None:
            return []
        return [(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in detecciones]

    def _recortar_cara(self, imagen_bgr, bbox, margen=0.2):
        x, y, w, h = bbox
        alto_img, ancho_img = imagen_bgr.shape[:2]
        mx, my = int(w * margen), int(h * margen)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(ancho_img, x + w + mx)
        y2 = min(alto_img, y + h + my)
        return imagen_bgr[y1:y2, x1:x2]

    def predecir(self, ruta_imagen):
        """Recibe ruta de imagen y devuelve la predicción."""
        imagen_bgr = cv2.imread(str(ruta_imagen))
        if imagen_bgr is None:
            raise RostroNoValido("No se pudo leer la imagen")
        return self._predecir_cv(imagen_bgr)

    @torch.no_grad()
    def predecir_bytes(self, imagen_bytes: bytes) -> dict:
        """Recibe bytes de imagen (JPEG/PNG) y devuelve la predicción."""
        arr = np.frombuffer(imagen_bytes, dtype=np.uint8)
        imagen_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if imagen_bgr is None:
            raise RostroNoValido("No se pudo decodificar la imagen")
        return self._predecir_cv(imagen_bgr)

    @torch.no_grad()
    def _predecir_cv(self, imagen_bgr):
        caras = self._detectar_caras(imagen_bgr)
        if caras:
            cara = self._recortar_cara(imagen_bgr, caras[0])
        else:
            cara = imagen_bgr

        imagen_rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
        imagen_pil = Image.fromarray(imagen_rgb)
        tensor = TRANSFORM(imagen_pil).unsqueeze(0).to(self.device)
        logits = self.modelo(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

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
            "caras_detectadas": len(caras),
        }
