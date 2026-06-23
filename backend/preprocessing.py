"""Pipeline de preprocesamiento de audio para inferencia (pasos 1-5 de Docs/pipeline_audio.md)."""

import io

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
TARGET_RMS = 0.1
FRAME_LENGTH = 2048
HOP_LENGTH = 512
TRIM_TOP_DB = 30.0
MIN_DURACION_S = 0.5


def preprocesar(audio_bytes: bytes) -> np.ndarray:
    """Recibe bytes de audio crudo y devuelve waveform float32 listo para HuBERT.

    Aplica: decodificación → mono → resample 16kHz → RMS norm → trim silencio.
    """
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)

    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

    rms = np.sqrt(np.mean(data ** 2))
    if rms >= 1e-8:
        data = (data * (TARGET_RMS / rms)).astype(np.float32)

    data = _trim_silencio(data)

    duracion = len(data) / SAMPLE_RATE
    if duracion < MIN_DURACION_S:
        raise ValueError(
            f"Audio demasiado corto tras el trim ({duracion:.2f} s). "
            "Sube un clip de al menos 0.5 s con contenido de voz."
        )

    return data


def _trim_silencio(data: np.ndarray) -> np.ndarray:
    if len(data) < FRAME_LENGTH:
        return data

    n_frames = 1 + (len(data) - FRAME_LENGTH) // HOP_LENGTH
    rms = np.array([
        np.sqrt(np.mean(data[i * HOP_LENGTH: i * HOP_LENGTH + FRAME_LENGTH] ** 2))
        for i in range(n_frames)
    ])

    ref = rms.max()
    if ref <= 0:
        return data

    db = 20 * np.log10(np.maximum(rms, 1e-10) / ref)
    activos = np.nonzero(db > -TRIM_TOP_DB)[0]
    if len(activos) == 0:
        return data

    inicio = int(activos[0] * HOP_LENGTH)
    fin = int(min(len(data), activos[-1] * HOP_LENGTH + FRAME_LENGTH))
    return data[inicio:fin]
