# Módulo de Voz — Guía General

**Proyecto:** PIA — NeuroEmoInnovat  
**Modelo base:** HuBERT (`facebook/hubert-base-ls960`) + cabeza clasificadora propia  
**Tarea:** clasificación de emoción a partir de audio de voz  
**Salida:** una de 7 emociones con distribución de probabilidades

---

## ¿Qué hace este módulo?

Recibe un archivo de audio con una locución y devuelve la emoción que expresa la voz. No analiza el contenido de las palabras (eso es el módulo de texto), sino los patrones acústicos: tono, ritmo, energía y timbre de la voz.

---

## Arquitectura del modelo

El módulo usa una arquitectura de dos piezas encadenadas:

```
Audio (wav)
    │
    ▼
┌─────────────────────────────────────────┐
│  HuBERT base  (facebook/hubert-base-ls960)  │
│  Transformer pre-entrenado en inglés    │
│  Congelado — solo extrae embeddings     │
│  Salida: vector float32 de 768 dims     │
└───────────────────┬─────────────────────┘
                    │  mean pooling temporal
                    ▼
┌─────────────────────────────────────────┐
│  CabezaEmocion  (entrenada por nosotros)    │
│  Dropout(0.3) → Linear(768→256) → ReLU  │
│  → Dropout(0.3) → Linear(256→7)         │
│  → Softmax → 7 probabilidades           │
└─────────────────────────────────────────┘
```

**HuBERT** es un modelo transformer pre-entrenado por Meta para representar audio de voz. Se usa completamente congelado — ninguno de sus parámetros se modifica durante el entrenamiento. Actúa como extractor de características: convierte el audio en un vector denso de 768 dimensiones que captura el contenido emocional de la voz.

**CabezaEmocion** es la única red que se entrenó. Es una red densa pequeña (dos capas lineales con dropout) que toma el vector de HuBERT y lo mapea a las 7 clases de emoción.

---

## Dataset de entrenamiento

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

| Detalle | Valor |
|---|---|
| Actores | 24 actores profesionales (12 hombres, 12 mujeres) |
| Archivos originales | ~1 440 clips de voz |
| Frecuencia de muestreo original | 48 000 Hz |
| Duración típica por clip | 3–5 s |

### Taxonomía unificada de 7 emociones

RAVDESS tiene 8 etiquetas originales. La emoción "calmado" se fusionó con "neutral" para producir 7 clases:

| Índice | Etiqueta interna | Español | Código RAVDESS |
|---|---|---|---|
| 0 | `neutral` | Neutral | 01 + 02 (fusionados) |
| 1 | `joy` | Felicidad | 03 |
| 2 | `sadness` | Tristeza | 04 |
| 3 | `anger` | Enojo | 05 |
| 4 | `fear` | Miedo | 06 |
| 5 | `disgust` | Disgusto | 07 |
| 6 | `surprise` | Sorpresa | 08 |

### Preprocesamiento del dataset

Antes de extraer embeddings se aplicó sobre cada clip:

1. Conversión a mono
2. Resampling a 16 000 Hz (requerido por HuBERT)
3. Normalización de RMS a `TARGET_RMS = 0.1` (iguala el volumen entre actores)
4. Recorte de silencio de inicio y fin (`TRIM_TOP_DB = 30 dB`)

---

## Entrenamiento

- El split se hizo **por actor** (no por clip) para evitar que la voz de un actor aparezca en entrenamiento y en evaluación al mismo tiempo. 70% entrenamiento / 15% validación / 15% test, estratificado por género.
- HuBERT se corrió una sola vez sobre todo el dataset para extraer los embeddings y guardarlos en caché. El entrenamiento de la cabeza opera sobre esos vectores directamente, sin volver a pasar audio por HuBERT.
- **Hiperparámetros:** 80 épocas, batch 32, Adam con lr=1e-3 y weight decay=1e-4.
- **Métrica de selección:** F1 macro en validación. Se guarda el checkpoint del mejor epoch.

---

## Inferencia

Para predecir sobre audio nuevo se aplica el mismo preprocesamiento del entrenamiento antes de pasar el audio al modelo. El pipeline completo está documentado en [pipeline_audio.md](pipeline_audio.md).

Resumen del flujo:

```
Audio crudo del usuario
    → Decodificar a PCM float32
    → Convertir a mono
    → Resamplear a 16 000 Hz
    → Normalizar RMS
    → Recortar silencio
    → HuBERT (congelado) → vector [768]
    → CabezaEmocion → softmax → 7 probabilidades
    → emoción predicha + confianza
```

**Restricciones importantes:**
- El audio debe tener contenido de voz real; clips de silencio o ruido producen predicciones sin sentido.
- El modelo fue entrenado con locuciones cortas (3–5 s). Clips muy largos (> 30 s) producen resultados menos fiables.
- HuBERT fue pre-entrenado en inglés. El modelo puede funcionar con otros idiomas pero no fue evaluado fuera del inglés/RAVDESS.

---

## API REST

El backend expone dos endpoints (FastAPI, `backend/main.py`):

### `GET /test`

Corre inferencia sobre el audio de prueba fijo (`dataset/Actor_01/03-01-03-01-01-01-01.wav`, emoción `joy`). Útil para verificar que el servidor arrancó correctamente.

```bash
curl http://localhost:8000/test
```

### `POST /predecir`

Recibe cualquier archivo de audio y devuelve la predicción.

```bash
curl -X POST http://localhost:8000/predecir \
  -F "audio=@mi_audio.wav"
```

**Respuesta:**

```json
{
  "emocion": "joy",
  "emocion_es": "felicidad",
  "confianza": 0.87,
  "ranking": [
    ["joy",      0.87],
    ["neutral",  0.06],
    ["sadness",  0.04],
    ["surprise", 0.01],
    ["anger",    0.01],
    ["fear",     0.01],
    ["disgust",  0.00]
  ]
}
```

---

## Estructura de archivos

```
backend/
├── main.py                        ← API FastAPI (endpoints + lifespan)
├── preprocessing.py               ← pipeline de audio pasos 1-5
├── requirements.txt
└── models/
    ├── voz/
    │   └── clasificador_voz.pt    ← checkpoint del modelo (pesos entrenados)
    └── src/
        ├── config.py              ← constantes del pipeline
        ├── modelo.py              ← carga HuBERT congelado
        ├── embeddings.py          ← carga audio + extrae embedding
        ├── clasificador.py        ← arquitectura CabezaEmocion (CRÍTICO)
        └── inferencia.py          ← clase Predictor (punto de entrada)

Docs/
├── modulo_voz.md                  ← este documento
└── pipeline_audio.md              ← guía técnica detallada del preprocesamiento
```

El archivo más crítico es `clasificador.py`: define la arquitectura de `CabezaEmocion`. Si se modifica, el checkpoint `clasificador_voz.pt` deja de cargar correctamente. Para más detalles sobre cada archivo de `src/` ver [../backend/models/src/guia_archivos_inferencia.md](../backend/models/src/guia_archivos_inferencia.md).

---

## Dónde leer más

| Documento | Contenido |
|---|---|
| [pipeline_audio.md](pipeline_audio.md) | Cada paso del preprocesamiento con código y explicación del por qué |
| [../backend/models/src/guia_archivos_inferencia.md](../backend/models/src/guia_archivos_inferencia.md) | Qué hace cada archivo de `src/`, qué se puede cambiar y qué no |
