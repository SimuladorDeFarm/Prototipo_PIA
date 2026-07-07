# PIA — Prototipo Multimodal de Prediccion de Emociones

## Que es este proyecto

Sistema de inteligencia artificial para predecir emociones a partir de senales multimodales (voz, texto, rostro). Recibe audio, imagen o texto y devuelve la emocion predicha sobre 7 clases.

## Arquitectura

```
frontend/          ← Vanilla HTML/CSS/JS (sin frameworks)
backend/           ← FastAPI (Python)
  main.py          ← punto de entrada de la API
  preprocessing.py ← pipeline de audio (mono, 16kHz, RMS, trim)
  models/
    voz/
      clasificador_voz_v4.pt      ← HuBERT fine-tuned v4 (state_dict completo)
    rostro/
      mejor_modelo_v2.pt          ← EfficientNet-B0 v2 (state_dict)
      face_detection_yunet_2023mar.onnx  ← detector facial (se descarga solo)
    texto/
      clasificador_texto_v4.pt    ← BETO fine-tuned v4 (state_dict)
    src/
      inferencia.py        ← Predictor de voz (HuBERTEmotionModel integrado)
      inferencia_rostro.py ← PredictorRostro (EfficientNet-B0 + YuNet)
      inferencia_texto.py  ← PredictorTexto (BETO + AutoModelForSequenceClassification)
Docs/
  modulo_voz.md
  modulo_rostro.md
  pipeline_audio.md
  pipeline_rostro.md
  pipeline_texto.md
```

## Modelos disponibles

| Modulo | Arquitectura | Checkpoint | F1 test |
|---|---|---|---|
| Voz | HuBERT (4 capas descongeladas) + cabeza 768→512→128→7 | `clasificador_voz_v4.pt` | ~0.744 |
| Rostro | EfficientNet-B0 + cabeza 1280→512→128→7 | `mejor_modelo_v2.pt` | ~0.640 |
| Texto | BETO full fine-tune + Focal Loss | `clasificador_texto_v4.pt` | ~0.166 |

## Pipeline de inferencia de voz

1. Decodificar audio a PCM float32
2. Convertir a mono
3. Resamplear a 16 000 Hz
4. Normalizar RMS a 0.1
5. Recortar silencio (TRIM_TOP_DB = 30 dB)
6. Pad/truncar a 48000 muestras (3s)
7. Forward pass por HuBERTEmotionModel (HuBERT + cabeza integrados) → softmax → 7 probabilidades

## Pipeline de inferencia de rostro

1. Decodificar imagen (JPEG/PNG)
2. Detectar cara con YuNet (umbral 0.5)
3. Recortar cara con 20% de margen (fallback: imagen completa si no detecta)
4. Resize a 224x224
5. Normalizar con media/std de ImageNet
6. Forward pass por EfficientNet-B0 → softmax → 7 probabilidades

## Pipeline de inferencia de texto

1. Tokenizar con BETO cased (max_length=128, padding="max_length")
2. Forward pass por AutoModelForSequenceClassification → softmax → 7 probabilidades

## Reglas de integracion

- **Los checkpoints guardan `state_dict()`**, no objetos completos. Cada modulo de inferencia reconstruye la arquitectura y carga los pesos.
- **HuBERT y BETO se descargan de HuggingFace** la primera vez y se cachean en `~/.cache/huggingface/`.
- **YuNet se descarga automaticamente** (~230KB) la primera vez que se usa el modulo de rostro.
- Los tres modulos son **independientes**: si falta un checkpoint, ese modulo queda deshabilitado (503) pero los demas funcionan.

## Levantar el backend

```bash
cd backend
.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Estado actual

- Los tres modulos (voz, rostro, texto) estan integrados y funcionales.
- La fusion multimodal combina predicciones por voto suave ponderado por F1.
- El frontend muestra los tres modulos lado a lado con la fusion al final.
