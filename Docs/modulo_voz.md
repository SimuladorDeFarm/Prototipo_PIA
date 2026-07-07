# Modulo de Voz — Guia General

**Proyecto:** PIA — NeuroEmo
**Modelo:** HuBERT fine-tuned v4 (`facebook/hubert-base-ls960`) con 4 capas descongeladas + cabeza clasificadora profunda
**Tarea:** clasificacion de emocion a partir de audio de voz
**Salida:** una de 7 emociones con distribucion de probabilidades

---

## Que hace este modulo

Recibe un archivo de audio con una locucion y devuelve la emocion que expresa la voz. No analiza el contenido de las palabras (eso es el modulo de texto), sino los patrones acusticos: tono, ritmo, energia y timbre de la voz.

---

## Arquitectura del modelo (v4)

El modulo usa un modelo integrado `HuBERTEmotionModel` que combina HuBERT con una cabeza clasificadora en un solo `nn.Module`:

```
Audio (wav, 3s max, 16kHz)
    │
    ▼
┌─────────────────────────────────────────┐
│  HuBERT base  (facebook/hubert-base-ls960)  │
│  Ultimas 4 capas descongeladas          │
│  (fine-tuned junto con la cabeza)       │
│  Salida: secuencia temporal [1, F, 768] │
└───────────────────┬─────────────────────┘
                    │  mean pooling temporal → [768]
                    ▼
┌─────────────────────────────────────────┐
│  Cabeza clasificadora profunda          │
│  Dropout(0.4) → Linear(768→512)         │
│  → BatchNorm1d(512) → ReLU             │
│  → Dropout(0.3) → Linear(512→128)       │
│  → ReLU → Dropout(0.2)                 │
│  → Linear(128→7) → Softmax             │
└─────────────────────────────────────────┘
```

### Diferencias con v1

| Aspecto | v1 | v4 (actual) |
|---|---|---|
| HuBERT | 100% congelado (solo extractor) | 4 capas descongeladas (fine-tuned) |
| Cabeza | 768→256→7 (simple) | 768→512→128→7 (con BatchNorm) |
| Checkpoint | Solo pesos de la cabeza + metadata | `state_dict()` del modelo completo |
| Data augmentation | No | Noise, gain, time shift |
| Scheduler | Adam fijo | Cosine warmup + discriminative LR |
| F1 test | 0.678 | ~0.744 |

---

## Dataset de entrenamiento

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

| Detalle | Valor |
|---|---|
| Actores | 24 actores profesionales (12 hombres, 12 mujeres) |
| Archivos | ~1 440 clips de voz |
| Frecuencia de muestreo | 16 000 Hz (resampleado desde 48 kHz) |
| Duracion tipica | 3–5 s |

### Taxonomia unificada de 7 emociones

| Indice | Etiqueta | Espanol | Codigo RAVDESS |
|---|---|---|---|
| 0 | `neutral` | Neutral | 01 + 02 (fusionados) |
| 1 | `joy` | Felicidad | 03 |
| 2 | `sadness` | Tristeza | 04 |
| 3 | `anger` | Enojo | 05 |
| 4 | `fear` | Miedo | 06 |
| 5 | `disgust` | Disgusto | 07 |
| 6 | `surprise` | Sorpresa | 08 |

---

## Entrenamiento (v4)

- Split **por actor** (70% train / 15% val / 15% test), estratificado por genero.
- **Hiperparametros:** 30 epocas, batch 16, AdamW con LR discriminativo (backbone 1e-5, cabeza 5e-4), cosine warmup (3 epocas), label smoothing 0.1, patience 8.
- **Data augmentation:** noise injection (SNR ~25-35 dB), gain perturbation (0.8-1.2x), time shift (hasta 10%).
- **Audio:** pad/truncar a 48000 muestras (3s a 16kHz) para batching uniforme.
- **Metrica de seleccion:** F1 macro en validacion.

---

## Inferencia

Pipeline completo documentado en [pipeline_audio.md](pipeline_audio.md).

```
Audio crudo del usuario
    → Decodificar a PCM float32
    → Convertir a mono
    → Resamplear a 16 000 Hz
    → Normalizar RMS (TARGET_RMS = 0.1)
    → Recortar silencio (TRIM_TOP_DB = 30 dB)
    → Pad/truncar a 48000 muestras
    → HuBERTEmotionModel (forward pass integrado)
    → softmax → 7 probabilidades
    → emocion predicha + confianza
```

---

## API REST

### `POST /predecir`

Recibe un archivo de audio y devuelve la prediccion.

```bash
curl -X POST http://localhost:8000/predecir -F "audio=@mi_audio.wav"
```

**Respuesta:**

```json
{
  "emocion": "joy",
  "emocion_es": "felicidad",
  "confianza": 0.87,
  "ranking": [["joy", 0.87], ["neutral", 0.06], ...]
}
```

---

## Estructura de archivos

```
backend/
├── main.py                        ← API FastAPI
├── preprocessing.py               ← pipeline de audio (pasos 1-5)
└── models/
    ├── voz/
    │   └── clasificador_voz_v4.pt ← checkpoint v4 (~362 MB)
    └── src/
        └── inferencia.py          ← HuBERTEmotionModel + Predictor
```
