# Guía de los Archivos de Inferencia — Todos los Módulos (v4)
## Para quien construye o mantiene el prototipo

> Este documento explica qué hace cada archivo de `src/` en el prototipo
> actual. Refleja la arquitectura v4 (modelos integrados).

---

## Mapa de archivos

```
backend/models/src/
├── inferencia.py          ← Predictor de voz (HuBERTEmotionModel v4)
├── inferencia_rostro.py   ← PredictorRostro (EfficientNet-B0 + YuNet)
├── inferencia_texto.py    ← PredictorTexto (BETO v4)
└── guia_archivos_inferencia.md  ← este documento
```

### Archivos legacy (no usados por v4)

Los siguientes archivos pertenecen a la v1 del módulo de voz y **no son
utilizados** por el código de inferencia actual:

```
clasificador.py   ← CabezaEmocion v1 (768→256→7), reemplazada por la cabeza integrada en HuBERTEmotionModel
config.py         ← constantes v1 (HIDDEN_DIM=256, etc.)
embeddings.py     ← extracción de embeddings separada (v1)
modelo.py         ← carga de HuBERT congelado por separado (v1)
```

---

## `inferencia.py` — Módulo de Voz (v4)

Define dos clases:

### `HuBERTEmotionModel(nn.Module)`

Modelo integrado que combina HuBERT + cabeza clasificadora en un solo `nn.Module`:

```
Audio [1, 48000]
    → HuBERT (4 últimas capas descongeladas)
    → mean pooling temporal → [768]
    → Dropout(0.4) → Linear(768→512) → BN → ReLU
    → Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.2)
    → Linear(128→7) → logits
```

El checkpoint `clasificador_voz_v4.pt` guarda el `state_dict()` completo
de este modelo (HuBERT + cabeza juntos, ~362 MB).

### `Predictor`

Wrapper que carga el modelo una vez y expone `predecir(ruta_audio)`:
1. Lee el audio con soundfile
2. Preprocesa (mono, 16kHz, RMS norm, trim silencio)
3. Pad/truncar a 48000 muestras
4. Forward pass → softmax → ranking de emociones

---

## `inferencia_rostro.py` — Módulo de Rostro (v2)

Define `PredictorRostro` y la excepción `RostroNoValido`.

### `PredictorRostro`

1. Carga EfficientNet-B0 (`torchvision.models.efficientnet_b0`) y reemplaza
   el clasificador con la cabeza entrenada (1280→512→128→7 con BatchNorm).
2. Carga el detector facial YuNet (ONNX, ~230 KB, se descarga automáticamente).
3. Expone `predecir_bytes(imagen_bytes)`:
   - Decodifica imagen (JPEG/PNG) con OpenCV
   - Detecta cara con YuNet (umbral 0.5)
   - Recorta cara con 20% margen (fallback: imagen completa)
   - Resize 224×224, normaliza con ImageNet stats
   - Forward pass → softmax → ranking

Checkpoint: `mejor_modelo_v2.pt` (~18 MB).

---

## `inferencia_texto.py` — Módulo de Texto (v4)

Define `PredictorTexto`.

### `PredictorTexto`

1. Carga BETO (`dccuchile/bert-base-spanish-wwm-cased`) con
   `AutoModelForSequenceClassification` (7 etiquetas).
2. Carga el `state_dict` desde `clasificador_texto_v4.pt` (~419 MB).
3. Expone `predecir(texto)`:
   - Tokeniza con max_length=128, padding="max_length"
   - Forward pass → softmax → ranking

Etiquetas: `others`(0), `joy`(1), `sadness`(2), `anger`(3), `fear`(4),
`disgust`(5), `surprise`(6).

---

## Checkpoints

| Módulo | Archivo | Tamaño | Contenido |
|---|---|---|---|
| Voz | `models/voz/clasificador_voz_v4.pt` | ~362 MB | `state_dict()` de HuBERTEmotionModel completo |
| Rostro | `models/rostro/mejor_modelo_v2.pt` | ~18 MB | `state_dict()` de EfficientNet-B0 + cabeza |
| Texto | `models/texto/clasificador_texto_v4.pt` | ~419 MB | `state_dict()` de BETO + cabeza clasificadora |

Todos guardan `state_dict()`, no objetos completos. Cada módulo de inferencia
reconstruye la arquitectura y luego carga los pesos.

---

*Documento actualizado el 2026-07-07. Refleja la arquitectura v4 del prototipo.*
