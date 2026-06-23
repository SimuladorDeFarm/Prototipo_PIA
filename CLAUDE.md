# PIA — Prototipo Multimodal de Predicción de Emociones

## Qué es este proyecto

Sistema de inteligencia artificial para predecir emociones a partir de señales multimodales (voz, texto, rostro). El objetivo actual es un prototipo funcional que recibe un archivo de audio y devuelve la emoción predicha. Los módulos de texto y rostro se integrarán en iteraciones posteriores.

## Arquitectura

```
frontend/          ← Vanilla HTML/CSS/JS (sin frameworks)
backend/           ← FastAPI (Python)
  main.py          ← punto de entrada de la API (actualmente esqueleto)
  models/
    clasificador_voz.pt        ← checkpoint del modelo entrenado
    src/
      config.py        ← todas las constantes y rutas del pipeline
      modelo.py        ← carga HuBERT congelado (extractor de embeddings)
      embeddings.py    ← carga audio + extrae embedding [768]
      clasificador.py  ← arquitectura CabezaEmocion (CRÍTICO: no modificar)
      inferencia.py    ← clase Predictor, punto de entrada de alto nivel
Docs/
  pipeline_audio.md  ← preprocesamiento completo de audio para inferencia web
  pipeline_rostro.md ← pipeline del módulo de rostro (pendiente integrar)
  pipeline_texto.md  ← pipeline del módulo de texto (pendiente integrar)
```

## Modelos disponibles

| Módulo | Estado | Checkpoint |
|---|---|---|
| Voz | Entrenado con fine-tuning | `backend/models/clasificador_voz.pt` |
| Texto | Entrenado | pendiente integrar |
| Rostro | Entrenado | pendiente integrar |

## Pipeline de inferencia de voz

El flujo completo está documentado en [Docs/pipeline_audio.md](Docs/pipeline_audio.md). Resumen:

1. Decodificar audio a PCM float32
2. Convertir a mono (promedio de canales)
3. Resamplear a 16 000 Hz
4. Normalizar RMS a `TARGET_RMS = 0.1`
5. Recortar silencio de inicio y fin (`TRIM_TOP_DB = 30 dB`)
6. Forward pass por HuBERT (`facebook/hubert-base-ls960`) + mean pooling → vector [768]
7. Cabeza clasificadora → softmax → 7 probabilidades

Emociones de salida: `neutral`, `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`.

## Reglas de integración del modelo

- **`CabezaEmocion` en `clasificador.py` no se modifica.** La arquitectura exacta (Dropout → Linear(768→256) → ReLU → Dropout → Linear(256→7)) debe coincidir con los pesos guardados en el `.pt`. Cambiarla rompe `load_state_dict`.
- **HuBERT se carga una sola vez al arrancar el servidor**, no por petición. El modelo pesa ~360 MB.
- **Mean pooling es obligatorio.** El clasificador fue entrenado con embeddings producidos por `last_hidden_state.mean(dim=1)`. Cambiar la operación de pooling produce predicciones incorrectas.
- El checkpoint es autosuficiente: contiene la arquitectura (`embedding_dim`, `hidden_dim`, `dropout`, `emociones`) además de los pesos.

Ver también [backend/models/src/guia_archivos_inferencia.md](backend/models/src/guia_archivos_inferencia.md) para detalles de cada módulo de `src/`.

## Cómo usar `Predictor` (clase de conveniencia)

```python
from backend.models.src.inferencia import Predictor

predictor = Predictor(ruta_checkpoint="backend/models/clasificador_voz.pt")
resultado = predictor.predecir("audio_preprocesado.wav")
# {
#   "emocion": "joy",
#   "emocion_es": "felicidad",
#   "confianza": 0.87,
#   "ranking": [("joy", 0.87), ("neutral", 0.06), ...]
# }
```

**Limitación:** `predecir()` asume que el audio ya está a 16 kHz. Para inferencia web hay que ejecutar los pasos 1–5 del pipeline antes de llamar al modelo, o modificar `predecir()` para aceptar un waveform en memoria.

## Levantar el backend

```bash
cd backend
uv run fastapi dev main.py
# o con uvicorn directamente:
uv run uvicorn main:app --reload
```

Requiere Python gestionado por `uv` (ver `.python-version`). Dependencias en `requirements.txt`.

## Estado actual

- `backend/main.py` es un esqueleto vacío con un único endpoint GET `/`. La integración del modelo de voz aún no está implementada.
- El frontend (`frontend/index.html`) existe pero no tiene lógica conectada al backend todavía.
- El prototipo funcional a construir: endpoint POST en FastAPI que reciba un `.wav`, ejecute el pipeline completo y devuelva la emoción predicha; frontend que suba el archivo y muestre el resultado.
