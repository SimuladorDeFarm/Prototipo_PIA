# PIA — Predicción de Emociones (Voz + Rostro)

Prototipo que predice la emoción a partir de **voz** (archivo o grabación) o de **rostro**
(foto subida o captura de cámara), sobre 7 clases: neutral, felicidad, tristeza, enojo,
miedo, disgusto y sorpresa. Incluye una API en FastAPI y un frontend web en vanilla JS
con los dos módulos lado a lado.

- **Voz:** HuBERT fine-tuneado + cabeza clasificadora (`clasificador_voz.pt`).
- **Rostro:** Py-Feat (Action Units) + Random Forest (`clasificador_rostro.joblib`).

Ambos módulos viven en el mismo backend y devuelven el mismo formato de respuesta.

---

## Requisitos

- **Python 3.12**
- ~360 MB de espacio para la caché de HuBERT (módulo de voz, se descarga la 1ª vez)
- Para el **módulo de rostro**:
  - Pesos de Py-Feat (se descargan automáticamente la 1ª vez, ~varios cientos de MB)
  - **FFmpeg** (DLLs *shared*) — necesario en Windows para `py-feat`/`torchcodec` (ver más abajo)
  - El archivo del modelo `clasificador_rostro.joblib` (no se versiona, ver más abajo)
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

### Colocar el modelo de rostro

El modelo `clasificador_rostro.joblib` pesa ~250 MB y **no se versiona** (supera el límite
de GitHub). Tras clonar, colócalo manualmente en:

```
backend/models/clasificador_rostro.joblib
```

Opciones para obtenerlo:
- Copiar el `.joblib` que te comparta el equipo, o
- Reentrenarlo tú con el script (requiere el CSV de features de Py-Feat):
  ```bash
  cd backend/training
  python entrenar_rostro.py --csv ruta/al/pyfeat_features_v5.csv
  ```

Sin este archivo, la voz funciona igual; el rostro queda deshabilitado (responde `503`).

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
