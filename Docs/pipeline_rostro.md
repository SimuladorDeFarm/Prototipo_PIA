# Pipeline de Inferencia — Módulo de Rostro

> **Propósito:** este documento describe, paso a paso, los procesos que debe
> atravesar una imagen antes de obtener una predicción de emoción del módulo de
> Rostro. El flujo es el **mismo** que se usó en entrenamiento (en las etapas que
> aplican a una sola imagen), por lo que cualquier desviación puede producir
> predicciones degradadas.
>
> **Modelo:** Random Forest entrenado sobre Action Units (FACS) extraídos con
> Py-Feat (Detectorv1).  
> **Entrada:** imagen con un rostro frontal (cualquier resolución, JPEG/PNG/BMP).  
> **Salida:** una de 7 emociones — `neutral`, `happy`, `sad`, `anger`, `fear`,
> `disgust`, `surprise`.
>
> **Diferencia con el pipeline de entrenamiento:** durante el entrenamiento, el
> dataset AffectNet partía de imágenes a 96×96 px que se reescalaron a 128×128 px
> antes de correr Py-Feat. En inferencia las imágenes de entrada ya tienen
> resolución alta (foto de cámara web, teléfono, etc.), por lo que ese paso de
> reescalado **no aplica**: Py-Feat recibe la imagen original y detecta el rostro
> independientemente de su resolución.

---

## Resumen del flujo

```
Imagen de entrada (cualquier resolución, 1 rostro)
   │
   ▼
1. Carga de la imagen
   │
   ▼
2. Extracción con Py-Feat (Detectorv1)
   │  → FaceScore, Pitch/Roll/Yaw (en RADIANES), 68 landmarks, 20 AUs
   │
   ▼
3. Validar FaceScore ≥ 0.90
   │  (si no se cumple → rechazar imagen)
   │
   ▼
4. Validar pose: |Yaw|, |Pitch|, |Roll| ≤ 45° (≈ 0.785 rad)
   │  (si no se cumple → rechazar imagen)
   │
   ▼
5. Validar landmarks: ningún valor NaN en los 136 valores (x_0..x_67, y_0..y_67)
   │  (si hay NaN → rechazar imagen)
   │
   ▼
6. Validar AUs: no todos en 0 ni todos en 1
   │  (si es degenerado → rechazar imagen)
   │
   ▼
7. Armar el vector de features: 20 AUs + FaceScore (21 valores, orden fijo)
   │
   ▼
8. Predecir con el Random Forest  →  emoción + probabilidades por clase
```

---

## Paso 1 — Carga de la imagen

El archivo de imagen se carga tal cual, sin ningún redimensionamiento previo. Py-Feat
acepta rutas a imágenes en disco en los formatos `.jpg`, `.jpeg`, `.png` y `.bmp`.

```python
ruta_imagen = "foto.jpg"   # ruta al archivo de entrada
```

**Qué NO se hace en inferencia (que sí se hizo en entrenamiento):**

- El dataset AffectNet venía a 96×96 px y se reescaló a 128×128 px antes de Py-Feat
  (tarea 5 del QC). En inferencia las imágenes llegan con resolución real: ese paso
  de reescalado no existe.
- No se aplican los filtros de luminancia/contraste ni de duplicados (son exclusivos
  del control de calidad del dataset).

---

## Paso 2 — Extracción de features con Py-Feat

Se crea un detector **Detectorv1** (el mismo modelo con el que se generó el dataset
de entrenamiento) y se corre sobre la imagen. Los modelos de identidad (ArcFace) y
mirada (gaze) se desactivan porque el proyecto no los usa.

```python
from feat import Detectorv1

detector = Detectorv1(
    device="cuda",           # o "cpu" si no hay GPU NVIDIA
    identity_model=None,     # desactivado: no se usa en este proyecto
    gaze_model=None,         # desactivado: no se usa en este proyecto
)

fex = detector.detect([ruta_imagen], data_type="image", progress_bar=False)
```

**Qué devuelve Py-Feat por cada imagen detectada:**

| Grupo | Columnas | Descripción |
|-------|----------|-------------|
| Bounding box | `FaceRectX`, `FaceRectY`, `FaceRectWidth`, `FaceRectHeight` | Rectángulo del rostro detectado |
| Confianza | `FaceScore` | Score de confianza de la detección (0–1) |
| Pose | `Pitch`, `Roll`, `Yaw` | Ángulos de la cabeza, en **RADIANES** |
| Landmarks | `x_0`..`x_67`, `y_0`..`y_67` | 68 puntos faciales, 136 valores |
| Action Units | `AU01`..`AU43` | 20 AUs del sistema FACS (probabilidades 0–1) |

> **Importante:** los ángulos de pose (`Pitch`, `Roll`, `Yaw`) vienen en **radianes**,
> no en grados. El umbral de 45° que se usa en la validación (paso 4) equivale a
> `π/4 ≈ 0.785 rad`. No comparar contra `45` directamente.

Si Py-Feat no detecta ningún rostro en la imagen, `FaceScore` viene como `NaN` y
la fila tiene `face_detected = False`. En ese caso la imagen se rechaza en el paso 3.

---

## Paso 3 — Validar FaceScore ≥ 0.90

El score de confianza de la detección debe ser **mayor o igual a 0.90**. Por debajo
de ese umbral la detección es poco fiable y los AUs calculados sobre ella no son
representativos de la expresión real.

```python
import numpy as np

facescore = fex["FaceScore"].values[0]

if np.isnan(facescore) or facescore < 0.90:
    raise ValueError(f"Rostro no detectado o confianza insuficiente (FaceScore={facescore:.3f} < 0.90).")
```

**Origen del umbral:** confirmado con el Líder Desarrollo, definido en
`src/config.py` (`FACESCORE_MIN = 0.90`). Se usó el mismo umbral para filtrar el
dataset de entrenamiento (tarea 7 del QC).

---

## Paso 4 — Validar pose: |Yaw|, |Pitch|, |Roll| ≤ 45°

Los tres ángulos de pose deben estar dentro del rango ±45° en valor absoluto. Un
rostro demasiado ladeado, inclinado hacia arriba/abajo o rotado hace que los Action
Units calculados sobre él no correspondan a la expresión real (Py-Feat estima AUs
sobre la proyección 2D del rostro, que se distorsiona con pose extrema).

**Los ángulos vienen en RADIANES.** El umbral de 45° se convierte:

```python
POSE_MAX_RAD = np.deg2rad(45.0)   # ≈ 0.7854 rad

yaw   = fex["Yaw"].values[0]
pitch = fex["Pitch"].values[0]
roll  = fex["Roll"].values[0]

if abs(yaw) > POSE_MAX_RAD or abs(pitch) > POSE_MAX_RAD or abs(roll) > POSE_MAX_RAD:
    raise ValueError(
        f"Pose extrema: Yaw={np.rad2deg(yaw):.1f}°, "
        f"Pitch={np.rad2deg(pitch):.1f}°, Roll={np.rad2deg(roll):.1f}°. "
        f"Límite: ±45°."
    )
```

**Origen del umbral:** ±45° (laxo), confirmado con el Líder Desarrollo. Definido en
`src/config.py` (`POSE_MAX_GRADOS = 45.0`). Mismo umbral que tarea 7 del QC.

---

## Paso 5 — Validar landmarks: sin NaN

Los 68 landmarks (136 coordenadas x/y) deben existir todos. La presencia de
coordenadas `NaN` indica que Py-Feat no pudo localizar esos puntos del rostro
con precisión, lo que generalmente acompaña a detecciones de baja calidad.

```python
LANDMARK_COLS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]

landmarks = fex[LANDMARK_COLS].values[0]

if np.isnan(landmarks).any():
    raise ValueError("Landmarks incompletos: hay coordenadas NaN en la detección.")
```

> Los landmarks no se usan como features de entrada del Random Forest (solo se usan
> los AUs). Esta validación descarta detecciones fallidas antes de calcular los AUs.

---

## Paso 6 — Validar AUs: no degenerados

Los 20 Action Units son probabilidades en el rango [0, 1]. Una fila donde **todos**
los AUs son exactamente 0 o **todos** son exactamente 1 indica que Py-Feat no
extrajo información real de la expresión (extracción fallida, no señal emocional).
Esas detecciones se descartan.

```python
AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43",
]

aus = fex[AU_COLS].values[0]

if (aus == 0).all() or (aus == 1).all():
    raise ValueError("AUs degenerados: todos en 0 o todos en 1 (extracción fallida).")
```

**Origen:** tarea 8 del QC. En el dataset v6 de entrenamiento no se encontró ninguna
fila que cumpliera esta condición, pero el filtro se mantiene en inferencia para
consistencia.

---

## Paso 7 — Armar el vector de features

El vector de entrada del Random Forest tiene exactamente **21 valores**, en el orden
exacto con que se entrenó el modelo:

1. Los 20 AUs (en el orden de `AU_COLS` del paso 6)
2. `FaceScore` (el score de confianza de la detección)

```python
FEATURE_COLS = AU_COLS + ["FaceScore"]   # 21 features

X = fex[FEATURE_COLS].values   # array shape (1, 21)
```

> **Por qué `FaceScore` va al final y no al principio:** así está definido en
> `src/modelo/datos.py` (`columnas_features()` = `AU_COLUMNS + RF_FEATURE_EXTRA`).
> El orden debe ser el mismo que el que vio el modelo en entrenamiento. Cambiarlo
> produce predicciones incorrectas silenciosamente.

---

## Paso 8 — Predicción con el Random Forest

Con el vector listo se carga el modelo y se predice:

```python
import joblib
import numpy as np

modelo = joblib.load("models/random_forest.joblib")

CLASES = ["neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"]

pred_clase  = modelo.predict(X)[0]              # emoción ganadora (str)
pred_probs  = modelo.predict_proba(X)[0]        # probabilidades (array 7 valores)

# Probabilidades por clase, en la taxonomía unificada del proyecto:
resultado = dict(zip(modelo.classes_, pred_probs))
print(f"Emoción: {pred_clase}  ({resultado[pred_clase]:.1%})")
```

**Nota sobre el orden de clases:** `modelo.classes_` devuelve las clases en el orden
en que el RF las aprendió (orden alfabético de las etiquetas del dataset), no
necesariamente el orden de la taxonomía unificada. Para la capa de fusión
multimodal, asegurarse de alinear por nombre de clase, no por índice de posición.

**Hiperparámetros del modelo entrenado** (ver `models/random_forest_meta.json`):

| Parámetro | Valor |
|-----------|-------|
| `n_estimators` | 300 |
| `max_features` | `sqrt` |
| `criterion` | `gini` |
| `bootstrap` | `True` |
| `class_weight` | `balanced` (precalculado como dict antes del fit) |
| `random_state` | 42 |

---

## Función de inferencia completa

Para uso directo en la aplicación web:

```python
import joblib
import numpy as np
from feat import Detectorv1

AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43",
]
LANDMARK_COLS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
FEATURE_COLS  = AU_COLS + ["FaceScore"]

FACESCORE_MIN = 0.90
POSE_MAX_RAD  = np.deg2rad(45.0)   # ≈ 0.785 rad


def inferir_emocion(
    ruta_imagen: str,
    detector: Detectorv1,
    modelo,
) -> dict:
    """
    Recibe una imagen, extrae AUs con Py-Feat y predice la emoción con el RF.

    Args:
        ruta_imagen: ruta al archivo (cualquier resolución, JPG/PNG/BMP).
        detector: instancia de Detectorv1 ya cargada (reutilizar entre llamadas).
        modelo: modelo Random Forest cargado con joblib.

    Returns:
        dict con 'emocion' (str) y 'probabilidades' (dict clase → float).

    Raises:
        ValueError si el rostro no pasa alguna de las validaciones.
    """
    fex = detector.detect([ruta_imagen], data_type="image", progress_bar=False)
    row = fex.iloc[0]

    # Validación 1: detección
    facescore = row["FaceScore"]
    if np.isnan(facescore) or facescore < FACESCORE_MIN:
        raise ValueError(f"FaceScore insuficiente ({facescore:.3f} < {FACESCORE_MIN}).")

    # Validación 2: pose (en radianes)
    for angulo in ("Yaw", "Pitch", "Roll"):
        val = row[angulo]
        if abs(val) > POSE_MAX_RAD:
            raise ValueError(
                f"Pose extrema en {angulo}: {np.rad2deg(val):.1f}° (límite ±45°)."
            )

    # Validación 3: landmarks sin NaN
    landmarks = row[LANDMARK_COLS].values.astype(float)
    if np.isnan(landmarks).any():
        raise ValueError("Landmarks incompletos (NaN detectados).")

    # Validación 4: AUs no degenerados
    aus = row[AU_COLS].values.astype(float)
    if (aus == 0).all() or (aus == 1).all():
        raise ValueError("AUs degenerados (todos en 0 o todos en 1).")

    # Vector de features (21 valores: 20 AUs + FaceScore)
    X = row[FEATURE_COLS].values.reshape(1, -1).astype(float)

    # Predicción
    pred_clase = modelo.predict(X)[0]
    pred_probs = modelo.predict_proba(X)[0]
    probs_dict = dict(zip(modelo.classes_, pred_probs.tolist()))

    return {"emocion": pred_clase, "probabilidades": probs_dict}


# ---------------------------------------------------------------------------
# Uso en la aplicación web (inicialización única al arrancar el servidor):
# ---------------------------------------------------------------------------
#
# detector = Detectorv1(device="cuda", identity_model=None, gaze_model=None)
# modelo   = joblib.load("models/random_forest.joblib")
#
# resultado = inferir_emocion("foto_usuario.jpg", detector, modelo)
# print(resultado)
# # → {'emocion': 'happy', 'probabilidades': {'anger': 0.02, 'disgust': 0.01, ...}}
```

---

## Consideraciones para imágenes capturadas en la web

Cuando la imagen proviene de una cámara web o de una foto subida por el usuario
(en lugar de un archivo de dataset preprocesado), hay aspectos adicionales a tener
en cuenta:

| Aspecto | Dataset (entrenamiento) | Captura web (inferencia) |
|---------|------------------------|--------------------------|
| Resolución | 96×96 px (AffectNet) → reescalado a 128×128 | Alta resolución (720p–4K típico) |
| Encuadre | Rostro centrado y recortado | Puede haber fondo, varias personas |
| Iluminación | Controlada (estudio) | Variable (ambiental) |
| Número de rostros | Siempre 1 | Puede haber más de 1 |

**Recomendaciones:**

1. **Múltiples rostros:** si Py-Feat detecta más de un rostro en la imagen
   (devuelve más de una fila), conservar solo la de mayor `FaceScore`:

   ```python
   fila = fex.sort_values("FaceScore", ascending=False).iloc[0:1]
   ```

2. **Iluminación extrema:** si el resultado tiene `FaceScore` consistentemente bajo
   (por ejemplo, contraluz), informar al usuario que mejore las condiciones de luz
   antes de capturar.

3. **No redimensionar antes de Py-Feat:** el redimensionado a 128×128 era necesario
   en entrenamiento para homogeneizar el dataset AffectNet (que venía a 96×96 px).
   En inferencia Py-Feat trabaja directamente sobre la resolución original; reducirla
   artificialmente antes puede degradar la calidad de la detección.

4. **Inicialización única del detector:** cargar el `Detectorv1` y el modelo RF una
   sola vez al arrancar el servidor y reutilizarlos en cada petición. Instanciarlos
   por petición añade ~2–5 segundos de latencia por carga de pesos.

---

## Dependencias

```txt
py-feat>=2.0.2   # Detectorv1 (arrastra PyTorch)
joblib>=1.3
numpy>=1.24
```

---

## Tabla resumen de umbrales

| Parámetro | Valor | Definido en |
|-----------|-------|-------------|
| Confianza mínima de detección | `FaceScore ≥ 0.90` | `src/config.py` → `FACESCORE_MIN` |
| Pose máxima (yaw/pitch/roll) | `≤ 45°` (≈ 0.785 rad) | `src/config.py` → `POSE_MAX_GRADOS` |
| Landmarks válidos | Ningún NaN en 136 coords | `src/qc/filtrado.py` tarea 8 |
| AUs válidos | No todos 0, no todos 1 | `src/qc/filtrado.py` tarea 8 |
| Features de entrada | 20 AUs + FaceScore = 21 | `src/modelo/datos.py` → `columnas_features()` |
| Clases de salida | 7 (taxonomía unificada) | `src/config.py` → `TAXONOMIA_7` |
