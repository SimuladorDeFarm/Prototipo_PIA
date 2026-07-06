# PIA — Predicción de Emociones (Voz + Rostro)

Prototipo que predice la emoción a partir de **voz** (archivo o grabación) o de **rostro**
(foto subida o captura de cámara), sobre 7 clases: neutral, felicidad, tristeza, enojo,
miedo, disgusto y sorpresa. Incluye una API en FastAPI y un frontend web en vanilla JS
con los dos módulos lado a lado.

- **Voz:** HuBERT fine-tuneado + cabeza clasificadora (`models/voz/clasificador_voz.pt`).
- **Rostro:** Py-Feat (Action Units) + Random Forest (`models/rostro/clasificador_rostro.joblib`).
- **Texto:** BETO fine-tuneado en EmoEvent (`models/texto/beto_emoevent_best.pth`).

Ambos módulos viven en el mismo backend y devuelven el mismo formato de respuesta.

---

## Requisitos

- **Python 3.12**
- ~360 MB de espacio para la caché de HuBERT (módulo de voz, se descarga la 1ª vez)
- Para el **módulo de rostro**:
  - Pesos de Py-Feat (se descargan automáticamente la 1ª vez, ~varios cientos de MB)
  - **FFmpeg** (DLLs *shared*) — necesario en Windows para `py-feat`/`torchcodec` (ver más abajo)
  - El archivo del modelo `models/rostro/clasificador_rostro.joblib` (no se versiona, ver [Modelos](#modelos))
- Un navegador moderno (para el frontend, micrófono y cámara)

---

## Instalación

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### FFmpeg en Windows (solo para el módulo de rostro)

`py-feat` arrastra `torchcodec`, que necesita las DLLs *shared* de **FFmpeg** (4–8) para
importarse. En Windows no vienen por defecto. Una vez tras instalar:

1. Descargar un build *shared* de FFmpeg 7 de
   [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
   (`ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip`).
2. Copiar las DLLs de su `bin/` (`avcodec-61.dll`, `avutil-59.dll`, `avformat-61.dll`,
   `swscale-8.dll`, etc.) dentro del paquete `torchcodec` del venv:
   `backend/.venv/Lib/site-packages/torchcodec/`.

Verificar: `python -c "from feat import Detectorv1; print('OK')"`.

> Si FFmpeg falta, el backend de **voz sigue funcionando** y el endpoint de rostro
> responde `503`. Más detalles en [Docs/modulo_rostro.md](Docs/modulo_rostro.md).

## Modelos

Los modelos están organizados **de forma modular**: cada módulo tiene su propia carpeta
dentro de `backend/models/`. El código lee cada modelo desde su carpeta, así que para
**cambiar un modelo por otro mejor (o peor) entrenado basta con reemplazar sus archivos**
en la carpeta correspondiente, sin tocar el código.

```
backend/models/
├── voz/
│   └── clasificador_voz.pt              ← módulo de voz
├── rostro/
│   ├── clasificador_rostro.joblib       ← módulo de rostro (modelo)
│   └── clasificador_rostro_meta.json    ← módulo de rostro (metadatos, opcional)
├── texto/
│   └── beto_emoevent_best.pth           ← módulo de texto
└── src/                                 ← código de inferencia (no tocar)
```

### Qué archivos necesita cada módulo

| Módulo | Carpeta | Archivo(s) necesario(s) para inferencia | Formato |
|---|---|---|---|
| **Voz** | `backend/models/voz/` | `clasificador_voz.pt` | Checkpoint PyTorch **autosuficiente**: contiene los pesos y la arquitectura (`embedding_dim`, `hidden_dim`, `dropout`, `emociones`). Es lo único que hace falta. |
| **Rostro** | `backend/models/rostro/` | `clasificador_rostro.joblib` | *Bundle* joblib **autodescriptivo**: `dict` con `modelo`, `features`, `clases`, `umbral_facescore`, `pose_max_grados`, `version`. Es lo único **requerido** para inferencia. |
| **Rostro** | `backend/models/rostro/` | `clasificador_rostro_meta.json` | Metadatos legibles (features, clases, hiperparámetros, métricas). **Opcional**: el backend no lo lee; solo documenta el modelo. Lo genera el script de entrenamiento junto al `.joblib`. |
| **Texto** | `backend/models/texto/` | `beto_emoevent_best.pth` | Checkpoint PyTorch con `model_state_dict` y `label2id`. BETO base se descarga solo desde HuggingFace (`dccuchile/bert-base-spanish-wwm-cased`). |

> **Importante:** solo debes reemplazar un modelo por **el mismo tipo de modelo** entrenado
> mejor o peor (p. ej. un `.joblib` nuevo de Random Forest para rostro, o un `.pt` nuevo de
> la misma cabeza clasificadora para voz). El archivo nuevo debe respetar el formato de la
> tabla, porque el código de `src/` espera esa estructura. Conserva el **mismo nombre de
> archivo** en la misma carpeta y reinicia el backend: el cambio se toma automáticamente.

### Cómo obtener / reemplazar cada modelo

- **Copiar el archivo** que te comparta un compañero (p. ej. si entrenó una versión mejor)
  en la carpeta indicada arriba, con el mismo nombre.
- **Reentrenar el de rostro** tú mismo (requiere el CSV de features de Py-Feat):
  ```bash
  cd backend/training
  python entrenar_rostro.py --csv ruta/al/pyfeat_features_v5.csv
  ```
  Esto escribe `clasificador_rostro.joblib` y `clasificador_rostro_meta.json` directamente
  en `backend/models/rostro/`.

### Qué se versiona y qué no

`clasificador_rostro.joblib` (~250 MB) y `beto_emoevent_best.pth` (~440 MB) **no se
versionan** (superan el límite de 100 MB de GitHub) y están en `.gitignore`. Tras clonar,
colócalos manualmente en su carpeta. El `clasificador_voz.pt` (~800 KB) y el `.json` de
metadatos sí se versionan.

Cada módulo es **independiente**: si falta un modelo (o sus dependencias), ese módulo queda
deshabilitado y su endpoint responde `503`, pero los demás siguen funcionando.

---

## Uso

El sistema tiene dos partes: el **backend** (API) y el **frontend** (web). Cada uno corre
en su propia terminal.

### 1. Levantar el backend

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Queda disponible en `http://localhost:8000`. En el log verás `Modelos listos.` y, si el
modelo de rostro está presente, `Modelo de rostro listo (bundle rostro-v5).`

> Alternativa: `fastapi dev main.py` (requiere instalar el extra `pip install "fastapi[standard]"`).

### 2. Levantar el frontend

En otra terminal:

```bash
cd frontend
python -m http.server 5500
```

Abre `http://localhost:5500` en el navegador.

> El frontend apunta a `http://localhost:8000` (en `frontend/js/app.js` y `frontend/js/rostro.js`).
> Si cambias el puerto del backend, ajústalo ahí.

### Desde el frontend puedes:

**Módulo de Voz (columna izquierda):**
- **Subir un archivo** `.wav` y predecir su emoción.
- **Grabar tu voz** con el micrófono (se convierte a WAV en el navegador).
- **Probar el audio de ejemplo** del dataset.

**Módulo de Rostro (columna derecha):**
- **Subir una imagen** (`.jpg`/`.png`) y predecir la emoción del rostro.
- **Usar la cámara** para capturar una foto y predecir.

Cada predicción muestra la emoción ganadora con su confianza y el ranking de las 7 clases.

---

## Uso directo de la API (opcional)

**Voz — audio de prueba fijo:**

```bash
curl http://localhost:8000/test
```

**Voz — subir tu propio audio:**

```bash
curl -X POST http://localhost:8000/predecir -F "audio=@ruta/al/archivo.wav"
```

**Rostro — subir una imagen:**

```bash
curl -X POST http://localhost:8000/predecir_rostro -F "imagen=@ruta/a/la/foto.jpg"
```

**Respuesta (ambos):**

```json
{
  "emocion": "happy",
  "emocion_es": "felicidad",
  "confianza": 0.87,
  "ranking": [["happy", 0.87], ["neutral", 0.06], ["sad", 0.03]]
}
```

Errores de validación del rostro (sin rostro detectado, pose extrema, etc.) → `422`.
Si el modelo de rostro no está instalado → `503`.

---

## Documentación

- [Docs/modulo_rostro.md](Docs/modulo_rostro.md) — módulo de rostro: estructura del modelo,
  cómo cambiarlo/reentrenarlo y setup de FFmpeg.
- [Docs/pipeline_rostro.md](Docs/pipeline_rostro.md) — pipeline detallado de inferencia de rostro.
- [Docs/pipeline_audio.md](Docs/pipeline_audio.md) — preprocesamiento de audio.
