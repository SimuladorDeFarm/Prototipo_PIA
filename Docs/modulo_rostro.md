# Módulo de Rostro (imagen) — Estructura del modelo y cómo cambiarlo

Este módulo predice la emoción de un rostro en una imagen. Espeja la estructura del
módulo de voz: un backend FastAPI con un endpoint que recibe el dato (aquí una
imagen) y un frontend que la envía y muestra el resultado.

- **Extracción de features:** Py-Feat `Detectorv1` → Action Units (FACS), pose, landmarks.
- **Clasificador:** un modelo de scikit-learn (por defecto Random Forest).
- **Clases (7):** `neutral, happy, sad, anger, fear, disgust, surprise`.

Pipeline detallado de validación e inferencia: ver [pipeline_rostro.md](pipeline_rostro.md).

---

## El contrato del modelo (lo importante)

El modelo **no** se guarda como un objeto "pelado". Se guarda como un **bundle
autodescriptivo** (`backend/models/clasificador_rostro.joblib`): un `dict` que
contiene tanto el modelo como el contrato que el backend necesita para usarlo.

```python
{
  "modelo":            <clasificador sklearn>,   # con .predict_proba(X) y .classes_
  "features":          ["AU01", ..., "AU43", "FaceScore"],  # columnas y ORDEN exacto
  "clases":            ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
  "umbral_facescore":  0.90,
  "pose_max_grados":   45.0,
  "version":           "rostro-v5"
}
```

`backend/models/src/inferencia_rostro.py` (`PredictorRostro`) lee **todo** desde este
bundle: qué columnas pedirle a Py-Feat y en qué orden, qué umbrales de calidad aplicar
y qué clases existen. **Nada de esto está hardcodeado en el backend.**

### Regla de intercambio

Puedes cambiar el modelo (reentrenarlo, usar otro algoritmo, otras features u otras
clases) y **el backend sigue funcionando sin tocar una línea**, siempre que el nuevo
artefacto cumpla:

1. Es un `dict` (bundle) con la clave `modelo` y, recomendado, `features`, `clases`,
   `umbral_facescore`, `pose_max_grados`.
2. `modelo` expone `.predict_proba(X)` y `.classes_` (cualquier clasificador sklearn:
   RandomForest, SVM con `probability=True`, GradientBoosting, LogisticRegression…).
3. Las columnas listadas en `features` son producibles por Py-Feat (AUs, `FaceScore`,
   pose, landmarks `x_i`/`y_i`).

Si el bundle declara otras `features` o `clases`, el backend se adapta solo.

---

## Cómo reentrenar / cambiar el modelo

El entrenamiento parte del CSV de features ya extraídos por Py-Feat
(`pyfeat_features_v5.csv`); **no requiere Py-Feat** (las features ya están calculadas).

```bash
python backend/training/entrenar_rostro.py --csv ruta/al/pyfeat_features_v5.csv
```

Esto regenera `backend/models/clasificador_rostro.joblib` (el bundle) y
`clasificador_rostro_meta.json` (metadatos legibles). Reinicia el backend y listo.

Para usar **otro algoritmo**, cambia el estimador en `entrenar_rostro.py` (la sección
`RandomForestClassifier(...)`) por cualquier clasificador sklearn con `predict_proba`.
El formato del bundle no cambia, así que la inferencia sigue igual.

---

## Estado actual del modelo entrenado

> ⚠️ El modelo incluido es un **prototipo funcional con precisión limitada**
> (~38% accuracy en Test). Causa: en el dataset de entrenamiento la clase `neutral`
> está muy subrepresentada (~360 ejemplos frente a miles de otras clases), porque solo
> aparece en la partición `Train_balanced`. El modelo aprende mal `neutral` y tiende a
> colapsar hacia `sad`.
>
> Esto **no afecta al funcionamiento del programa**: la arquitectura está pensada
> justo para que, con mejores datos o mejor balanceo, se reentrene y se reemplace el
> bundle sin tocar el backend. Calidad del modelo y funcionamiento están desacoplados.

---

## Instalación en Windows (FFmpeg)

Py-Feat 2.x arrastra `torchcodec`, que necesita las DLLs *shared* de **FFmpeg** (4–8)
para importarse, aunque solo usemos imágenes. En Windows no vienen por defecto, así
que tras `pip install -r requirements.txt` hay que añadirlas una vez:

1. Descargar un build *shared* de FFmpeg 7 para Windows (p. ej. de
   [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases):
   `ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip`).
2. Copiar las DLLs de su carpeta `bin/` (`avcodec-61.dll`, `avutil-59.dll`,
   `avformat-61.dll`, `swscale-8.dll`, etc.) dentro del paquete `torchcodec` del venv:
   `backend/.venv/Lib/site-packages/torchcodec/` (junto a `libtorchcodec_core7.dll`).

Verificar: `python -c "from feat import Detectorv1; print('OK')"`.

Si FFmpeg falta, `import feat` lanza `RuntimeError: Could not load libtorchcodec`.
En ese caso el backend de voz sigue funcionando y `/predecir_rostro` responde 503.

---

## Endpoint

`POST /predecir_rostro` — recibe un campo `imagen` (multipart) y devuelve:

```json
{
  "emocion": "happy",
  "emocion_es": "felicidad",
  "confianza": 0.88,
  "ranking": [["happy", 0.88], ["sad", 0.05], ...]
}
```

Errores de validación (rostro no detectado, pose extrema, etc.) → HTTP 422 con el
motivo. Si Py-Feat no está instalado o no hay modelo → HTTP 503 (el resto del backend,
incluido el módulo de voz, sigue funcionando).

Frontend: `frontend/rostro.html` (subir imagen o capturar con cámara).
