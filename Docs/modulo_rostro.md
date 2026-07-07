# Modulo de Rostro — Guia General

**Proyecto:** PIA — NeuroEmo
**Modelo:** EfficientNet-B0 fine-tuned sobre AffectNet + deteccion facial YuNet
**Tarea:** clasificacion de emocion a partir de imagen facial
**Salida:** una de 7 emociones con distribucion de probabilidades

---

## Que hace este modulo

Recibe una imagen con un rostro y devuelve la emocion que expresa la expresion facial. Usa deteccion de rostro automatica (YuNet) para recortar la cara antes de clasificar con EfficientNet-B0.

---

## Arquitectura del modelo (v2)

```
Imagen (JPEG/PNG, cualquier resolucion)
    │
    ▼
┌─────────────────────────────────────────┐
│  YuNet (face_detection_yunet_2023mar)   │
│  Detector facial ONNX (~230 KB)         │
│  Umbral de confianza: 0.5               │
│  Salida: bounding box (x, y, w, h)      │
│  Fallback: imagen completa si no detecta│
└───────────────────┬─────────────────────┘
                    │  recortar cara + 20% margen
                    ▼
┌─────────────────────────────────────────┐
│  Resize 224×224 + Normalize (ImageNet)  │
│  mean=[0.485, 0.456, 0.406]            │
│  std=[0.229, 0.224, 0.225]             │
└───────────────────┬─────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│  EfficientNet-B0 (completamente         │
│  descongelado, fine-tuned)              │
│  Features: 1280 dims                    │
└───────────────────┬─────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│  Cabeza clasificadora                   │
│  Dropout(0.4) → Linear(1280→512)        │
│  → BatchNorm1d(512) → ReLU             │
│  → Dropout(0.3) → Linear(512→128)       │
│  → ReLU → Dropout(0.2)                 │
│  → Linear(128→7) → Softmax             │
└─────────────────────────────────────────┘
```

---

## Dataset de entrenamiento

**AffectNet** — dataset de expresiones faciales en estado natural (in-the-wild).

| Detalle | Valor |
|---|---|
| Resolucion de entrenamiento | 224×224 px (resized desde ~96×96) |
| Clases | 7 emociones |
| Augmentation | Random erasing, affine, color jitter |

### Clases (7 emociones)

| Indice | Etiqueta | Espanol |
|---|---|---|
| 0 | `anger` | Enojo |
| 1 | `disgust` | Disgusto |
| 2 | `fear` | Miedo |
| 3 | `happy` | Alegria |
| 4 | `neutral` | Neutral |
| 5 | `sad` | Tristeza |
| 6 | `surprise` | Sorpresa |

---

## Entrenamiento (v2)

- EfficientNet-B0 **completamente descongelado** con LR discriminativo.
- Pesos iniciales: ImageNet pre-trained.
- Cosine annealing con warmup.
- Label smoothing en la loss.
- Data augmentation agresivo (random erasing, affine, color jitter).
- **Metrica de seleccion:** F1 macro en validacion.
- **F1 test:** ~0.640

---

## Deteccion facial (YuNet)

Se usa **OpenCV FaceDetectorYN** con el modelo YuNet (ONNX):
- Se descarga automaticamente (~230 KB) la primera vez.
- Umbral de confianza: 0.5
- Si no detecta ninguna cara, se usa la **imagen completa** como fallback.
- Si detecta multiples caras, se usa la primera (mayor confianza).

---

## Endpoint

`POST /predecir_rostro` — recibe un campo `imagen` (multipart) y devuelve:

```json
{
  "emocion": "happy",
  "emocion_es": "alegria",
  "confianza": 0.88,
  "ranking": [["happy", 0.88], ["neutral", 0.05], ...],
  "caras_detectadas": 1
}
```

Si no se puede decodificar la imagen → HTTP 422.
Si el modulo no esta disponible → HTTP 503.

---

## Estructura de archivos

```
backend/models/
├── rostro/
│   ├── mejor_modelo_v2.pt                    ← pesos EfficientNet-B0 (~18 MB)
│   └── face_detection_yunet_2023mar.onnx     ← YuNet (se descarga solo)
└── src/
    └── inferencia_rostro.py                  ← PredictorRostro
```
