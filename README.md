# PIA — Predicción de Emociones por Voz

Prototipo que recibe audio (archivo o grabación desde el micrófono) y predice la emoción usando un modelo HuBERT fine-tuneado sobre 7 clases: neutral, felicidad, tristeza, enojo, miedo, disgusto y sorpresa. Incluye una API en FastAPI y un frontend web en vanilla JS.

---

## Requisitos

- Python 3.12
- ~360 MB de espacio para la caché de HuBERT (se descarga automáticamente la primera vez)
- Un navegador moderno (para el frontend y la grabación por micrófono)

---

## Instalación

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Uso

El sistema tiene dos partes: el **backend** (API) y el **frontend** (web). Cada uno corre en su propia terminal.

### 1. Levantar el backend

```bash
cd backend
.venv/bin/fastapi dev main.py
```

Queda disponible en `http://localhost:8000`.

### 2. Levantar el frontend

En otra terminal:

```bash
cd frontend
python3 -m http.server 5500
```

Abre `http://localhost:5500` en el navegador.

> El frontend apunta a `http://localhost:8000` (definido en `frontend/js/app.js`). Si cambias el puerto del backend, ajústalo ahí.

### Desde el frontend puedes:

- **Subir un archivo** `.wav` y predecir su emoción.
- **Grabar tu voz** con el micrófono: se convierte a WAV en el navegador y se envía al backend.
- **Probar el audio de ejemplo** (un clip fijo del dataset), sin subir nada.

Cada predicción muestra la emoción ganadora con su confianza y el ranking completo de las 7 clases.

---

## Uso directo de la API (opcional)

Si prefieres consumir la API sin el frontend:

**Probar con el audio fijo de prueba:**

```bash
curl http://localhost:8000/test
```

**Subir tu propio audio:**

```bash
curl -X POST http://localhost:8000/predecir \
  -F "audio=@ruta/al/archivo.wav"
```

**Respuesta:**

```json
{
  "emocion": "joy",
  "emocion_es": "felicidad",
  "confianza": 0.87,
  "ranking": [["joy", 0.87], ["neutral", 0.06], ["sadness", 0.03]]
}
```
