# PIA — Predicción de Emociones por Voz

Prototipo de API que recibe un archivo de audio y predice la emoción usando un modelo HuBERT fine-tuneado sobre 7 clases: neutral, felicidad, tristeza, enojo, miedo, disgusto y sorpresa.

---

## Requisitos

- Python 3.12
- ~360 MB de espacio para la caché de HuBERT (se descarga automáticamente la primera vez)

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

**Levantar el servidor:**

```bash
cd backend
.venv/bin/fastapi dev main.py
```

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
