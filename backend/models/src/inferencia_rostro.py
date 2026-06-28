"""Inferencia: predecir la emoción de un rostro en una imagen.

Reúne las dos piezas del pipeline de rostro:
  1. Py-Feat (Detectorv1) — extrae Action Units (FACS), pose y landmarks.
  2. Un clasificador entrenado (Random Forest u otro) guardado como BUNDLE.

El bundle es un dict AUTODESCRIPTIVO (ver backend/training/entrenar_rostro.py):

    {
      "modelo":   <clasificador sklearn>,   # con .predict_proba(X) y .classes_
      "features": [...columnas y su ORDEN...],
      "clases":   [...emociones...],
      "umbral_facescore": 0.90,
      "pose_max_grados":  45.0,
      "version":  "rostro-v5",
    }

Esta clase lee TODO el contrato desde el bundle (qué columnas usar y en qué orden,
qué umbrales aplicar, qué clases existen). Por eso, para cambiar el modelo basta
con guardar otro bundle con este mismo formato: este código sigue funcionando sin
modificaciones, aunque cambie el tipo de clasificador, las features o las clases.

Es SOLO inferencia: no entrena ni modifica nada. La metodología de validación
(FaceScore, pose, landmarks, AUs) replica Docs/pipeline_rostro.md.
"""

from pathlib import Path

import joblib
import numpy as np

# Nombres de las emociones en español para mostrar al usuario.
NOMBRES_ES = {
    "neutral": "neutral",
    "happy": "felicidad",
    "sad": "tristeza",
    "anger": "enojo",
    "fear": "miedo",
    "disgust": "disgusto",
    "surprise": "sorpresa",
}

# Columnas que Py-Feat produce y que el pipeline valida (independiente del modelo).
LANDMARK_COLS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
POSE_COLS = ("Yaw", "Pitch", "Roll")


class RostroNoValido(ValueError):
    """La imagen no pasó alguna validación de calidad (rostro/pose/landmarks/AUs)."""


class PredictorRostro:
    """Carga Py-Feat + el clasificador una sola vez y predice sobre imágenes.

    Todo el contrato (features, clases, umbrales) se lee del bundle, no se
    hardcodea: cambiar el modelo = cambiar el bundle.
    """

    def __init__(self, ruta_bundle, device=None):
        ruta = Path(ruta_bundle)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe el modelo de rostro: {ruta}. "
                f"Entrena con backend/training/entrenar_rostro.py."
            )

        bundle = joblib.load(ruta)
        if not isinstance(bundle, dict) or "modelo" not in bundle:
            raise ValueError(
                f"Formato de bundle no válido en {ruta}: se esperaba un dict con la "
                f"clave 'modelo'. Reentrena con entrenar_rostro.py."
            )

        # --- Contrato leído desde el bundle ---
        self.modelo = bundle["modelo"]
        self.features = list(bundle["features"])
        self.clases = list(bundle.get("clases", getattr(self.modelo, "classes_", [])))
        self.umbral_facescore = float(bundle.get("umbral_facescore", 0.90))
        self.pose_max_rad = np.deg2rad(float(bundle.get("pose_max_grados", 45.0)))
        self.version = bundle.get("version", "desconocida")

        if not hasattr(self.modelo, "predict_proba"):
            raise ValueError(
                "El modelo del bundle no expone predict_proba(); no se puede usar "
                "para inferencia con ranking de probabilidades."
            )

        # --- Detector Py-Feat (carga única, ~2-5 s) ---
        from feat import Detectorv1  # import perezoso: py-feat es pesado

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.detector = Detectorv1(
            device=device,
            identity_model=None,   # no se usa en este proyecto
            gaze_model=None,       # no se usa en este proyecto
        )

    # ------------------------------------------------------------------ #
    def _seleccionar_rostro(self, fex):
        """De la salida de Py-Feat, conserva la fila del rostro de mayor FaceScore."""
        if fex is None or len(fex) == 0:
            raise RostroNoValido("No se detectó ningún rostro en la imagen.")
        return fex.sort_values("FaceScore", ascending=False).iloc[0]

    def _validar(self, row):
        """Aplica los filtros de calidad del pipeline (Docs/pipeline_rostro.md)."""
        facescore = row.get("FaceScore", np.nan)
        if np.isnan(facescore) or facescore < self.umbral_facescore:
            raise RostroNoValido(
                f"Rostro no detectado o confianza insuficiente "
                f"(FaceScore={facescore:.3f} < {self.umbral_facescore})."
            )

        for ang in POSE_COLS:
            val = float(row[ang])
            if abs(val) > self.pose_max_rad:
                raise RostroNoValido(
                    f"Pose extrema en {ang}: {np.rad2deg(val):.1f}° (límite ±"
                    f"{np.rad2deg(self.pose_max_rad):.0f}°)."
                )

        landmarks = row[LANDMARK_COLS].values.astype(float)
        if np.isnan(landmarks).any():
            raise RostroNoValido("Landmarks incompletos (coordenadas NaN).")

    def predecir(self, ruta_imagen):
        """Devuelve un dict con la emoción predicha y las probabilidades por clase.

        Formato idéntico al módulo de voz:
            {"emocion", "emocion_es", "confianza", "ranking"}
        """
        fex = self.detector.detect(
            [str(ruta_imagen)], data_type="image", progress_bar=False
        )
        row = self._seleccionar_rostro(fex)
        self._validar(row)

        # Vector de features EN EL ORDEN del contrato del bundle.
        try:
            x = row[self.features].values.reshape(1, -1).astype(float)
        except KeyError as e:
            raise RostroNoValido(
                f"Py-Feat no produjo la columna requerida por el modelo: {e}."
            )

        # AUs degenerados (validación específica de las features de Action Units).
        au_features = [c for c in self.features if c.startswith("AU")]
        if au_features:
            aus = row[au_features].values.astype(float)
            if (aus == 0).all() or (aus == 1).all():
                raise RostroNoValido("AUs degenerados (todos en 0 o todos en 1).")

        probs = self.modelo.predict_proba(x)[0]
        clases = list(self.modelo.classes_)
        ranking = sorted(
            ((clases[i], float(probs[i])) for i in range(len(clases))),
            key=lambda p: p[1], reverse=True,
        )
        emocion, confianza = ranking[0]
        return {
            "emocion": emocion,
            "emocion_es": NOMBRES_ES.get(emocion, emocion),
            "confianza": confianza,
            "ranking": ranking,
        }
