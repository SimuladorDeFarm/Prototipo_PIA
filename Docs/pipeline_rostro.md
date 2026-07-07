# Pipeline de Inferencia — Modulo de Rostro

> **Modelo:** EfficientNet-B0 fine-tuned sobre AffectNet + deteccion facial YuNet.
> **Entrada:** imagen con un rostro (cualquier resolucion, JPEG/PNG).
> **Salida:** una de 7 emociones — `anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

---

## Resumen del flujo

```
Imagen de entrada (cualquier resolucion)
   │
   ▼
1. Decodificar imagen (bytes → BGR con OpenCV)
   │
   ▼
2. Detectar cara con YuNet (umbral 0.5)
   │  → Si detecta: recortar cara con 20% margen
   │  → Si no detecta: usar imagen completa
   │
   ▼
3. Convertir BGR → RGB
   │
   ▼
4. Resize a 224×224 px
   │
   ▼
5. Normalizar con media/std de ImageNet
   │  mean=[0.485, 0.456, 0.406]
   │  std=[0.229, 0.224, 0.225]
   │
   ▼
6. Forward pass por EfficientNet-B0
   │
   ▼
7. Softmax → 7 probabilidades → emocion predicha
```

---

## Paso 1 — Decodificar imagen

La imagen llega como bytes (JPEG/PNG) desde el frontend. Se decodifica a un array BGR con OpenCV:

```python
arr = np.frombuffer(imagen_bytes, dtype=np.uint8)
imagen_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
```

---

## Paso 2 — Deteccion facial con YuNet

Se usa `cv2.FaceDetectorYN` con el modelo YuNet (ONNX):

```python
detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (ancho, alto))
detector.setScoreThreshold(0.5)
_, detecciones = detector.detect(imagen_bgr)
```

- Si detecta una o mas caras: se toma la primera y se recorta con 20% de margen alrededor del bounding box.
- Si no detecta ninguna cara: se usa la imagen completa como entrada al modelo.

El recorte con margen:

```python
x, y, w, h = bbox
mx, my = int(w * 0.2), int(h * 0.2)
cara = imagen[max(0,y-my):min(alto,y+h+my), max(0,x-mx):min(ancho,x+w+mx)]
```

---

## Paso 3 — Conversion BGR → RGB

OpenCV carga en BGR, pero el modelo fue entrenado con imagenes RGB (torchvision):

```python
imagen_rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
imagen_pil = Image.fromarray(imagen_rgb)
```

---

## Paso 4 — Resize a 224x224

```python
transforms.Resize((224, 224))
```

El modelo fue entrenado con imagenes redimensionadas a 224×224 px.

---

## Paso 5 — Normalizacion ImageNet

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

EfficientNet-B0 fue pre-entrenado con ImageNet; se usa la misma normalizacion.

---

## Paso 6 — Forward pass

```python
logits = modelo(tensor)  # [1, 7]
probs = F.softmax(logits, dim=1).squeeze(0)  # [7]
```

---

## Paso 7 — Interpretacion

Las 7 clases en orden:

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

## Consideraciones para imagenes de camara web

| Aspecto | Dataset (entrenamiento) | Captura web (inferencia) |
|---------|------------------------|--------------------------|
| Resolucion | ~96×96 (AffectNet) → resize 224×224 | Alta resolucion (720p+) |
| Encuadre | Rostro centrado | Puede haber fondo, varias personas |
| Iluminacion | Variable (in-the-wild) | Variable (ambiental) |

El paso de deteccion facial (YuNet) mitiga las diferencias de encuadre.

---

## Dependencias

```txt
torch
torchvision
opencv-python
Pillow
numpy
```
