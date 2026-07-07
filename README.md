# PIA — Predicción de Emociones (Voz + Rostro + Texto)

Prototipo multimodal que predice la emoción a partir de **voz**, **rostro** o **texto**,
sobre 7 clases: neutral, felicidad, tristeza, enojo, miedo, disgusto y sorpresa.
Incluye una API en FastAPI, un frontend web en vanilla JS con los tres módulos lado a lado,
y una fusión multimodal por voto suave ponderado.

![Interfaz web de PIA con los módulos de voz, rostro y texto](2026-07-06_16-18.png)

- **Voz:** HuBERT fine-tuned (4 capas descongeladas) + cabeza clasificadora profunda (`models/voz/clasificador_voz_v4.pt`).
- **Rostro:** EfficientNet-B0 fine-tuned + detección facial YuNet (`models/rostro/mejor_modelo_v2.pt`).
- **Texto:** BETO fine-tuned con Focal Loss en EMOEvent (`models/texto/clasificador_texto_v4.pt`).

Los tres módulos viven en el mismo backend y devuelven el mismo formato de respuesta.

---

## Requisitos

- **Python 3.11+**
- ~360 MB de espacio para la caché de HuBERT (módulo de voz, se descarga la 1a vez)
- ~440 MB para la caché de BETO (módulo de texto, se descarga la 1a vez)
- YuNet (~230 KB) se descarga automáticamente la 1a vez (módulo de rostro)
- Un navegador moderno (para el frontend, micrófono y cámara)

---

## Instalacion

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Modelos

Los modelos estan organizados de forma modular: cada modulo tiene su propia carpeta
dentro de `backend/models/`.

```
backend/models/
├── voz/
│   └── clasificador_voz_v4.pt        ← HuBERT fine-tuned v4 (~362 MB)
├── rostro/
│   ├── mejor_modelo_v2.pt            ← EfficientNet-B0 v2 (~18 MB)
│   └── face_detection_yunet_2023mar.onnx  ← YuNet (se descarga solo, ~230 KB)
├── texto/
│   └── clasificador_texto_v4.pt      ← BETO fine-tuned v4 (~419 MB)
└── src/                              ← codigo de inferencia
    ├── inferencia.py                 ← Predictor de voz (HuBERTEmotionModel)
    ├── inferencia_rostro.py          ← PredictorRostro (EfficientNet-B0 + YuNet)
    └── inferencia_texto.py           ← PredictorTexto (BETO)
```

### Que archivo necesita cada modulo

| Modulo | Checkpoint | Arquitectura | Dataset | F1 test |
|---|---|---|---|---|
| **Voz** | `clasificador_voz_v4.pt` | HuBERT (4 capas descongeladas) + cabeza 768→512→128→7 | RAVDESS | ~0.744 |
| **Rostro** | `mejor_modelo_v2.pt` | EfficientNet-B0 + cabeza 1280→512→128→7 | AffectNet | ~0.640 |
| **Texto** | `clasificador_texto_v4.pt` | BETO (full fine-tune) + clasificacion 7 clases | EMOEvent (es) | ~0.166 |

> **Importante:** los checkpoints `.pt` superan los 100 MB de GitHub y estan en `.gitignore`.
> Tras clonar, colocalos manualmente en su carpeta o descargalos desde Google Drive.

Cada modulo es **independiente**: si falta un modelo (o sus dependencias), ese modulo queda
deshabilitado y su endpoint responde `503`, pero los demas siguen funcionando.

---

## Uso

### 1. Levantar el backend

```bash
cd backend
.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Queda disponible en `http://localhost:8000`.

### 2. Levantar el frontend

En otra terminal:

```bash
cd frontend
python -m http.server 5501
```

Abre `http://localhost:5501` en el navegador.

### Desde el frontend puedes:

**Modulo de Voz:**
- Subir un archivo `.wav` y predecir su emocion.
- Grabar tu voz con el microfono (se convierte a WAV en el navegador).

**Modulo de Rostro:**
- Subir una imagen (`.jpg`/`.png`) y predecir la emocion del rostro.
- Usar la camara para capturar una foto y predecir.

**Modulo de Texto:**
- Escribir un texto en espanol y predecir la emocion.

**Fusion Multimodal:**
- Combinar las predicciones de los modulos disponibles mediante voto suave ponderado por F1.

---

## Uso directo de la API

**Voz:**
```bash
curl -X POST http://localhost:8000/predecir -F "audio=@archivo.wav"
```

**Rostro:**
```bash
curl -X POST http://localhost:8000/predecir_rostro -F "imagen=@foto.jpg"
```

**Texto:**
```bash
curl -X POST http://localhost:8000/predecir_texto -H "Content-Type: application/json" -d '{"texto": "Estoy muy feliz hoy"}'
```

**Respuesta (los tres modulos):**

```json
{
  "emocion": "joy",
  "emocion_es": "felicidad",
  "confianza": 0.87,
  "ranking": [["joy", 0.87], ["neutral", 0.06], ["sadness", 0.03], ...]
}
```

---

## Documentacion

- [Docs/modulo_voz.md](Docs/modulo_voz.md) — modulo de voz: arquitectura, dataset, entrenamiento.
- [Docs/modulo_rostro.md](Docs/modulo_rostro.md) — modulo de rostro: EfficientNet-B0 + YuNet.
- [Docs/pipeline_audio.md](Docs/pipeline_audio.md) — preprocesamiento de audio.
- [Docs/pipeline_rostro.md](Docs/pipeline_rostro.md) — pipeline de inferencia de rostro.
- [Docs/pipeline_texto.md](Docs/pipeline_texto.md) — preprocesamiento de texto para BETO.
