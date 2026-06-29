# Pipeline de Preprocesamiento de Audio — Inferencia Web
## Módulo de Voz · NeuroEmoInnovat (PIA)

> **Propósito de este documento.**  
> Instructivo técnico paso a paso que describe las transformaciones que debe
> recibir un archivo de audio antes de introducirse al modelo de inferencia del
> módulo de voz. El flujo aquí descrito replica fielmente el preprocesamiento
> aplicado durante el entrenamiento, para que la entrada en producción sea
> estadísticamente consistente con los datos que el modelo vio durante el ajuste.
>
> **Audiencia.** Equipo de desarrollo del prototipo web multimodal. Se asume
> acceso al backend de Python donde corren los modelos y conocimientos básicos de
> procesamiento de señales de audio.

---

## Índice

1. [Entradas soportadas](#1-entradas-soportadas)
2. [Resumen ejecutivo del pipeline](#2-resumen-ejecutivo-del-pipeline)
3. [Paso 1 — Decodificación del audio](#3-paso-1--decodificación-del-audio)
4. [Paso 2 — Conversión a mono](#4-paso-2--conversión-a-mono)
5. [Paso 3 — Resampling a 16 000 Hz](#5-paso-3--resampling-a-16-000-hz)
6. [Paso 4 — Normalización de RMS](#6-paso-4--normalización-de-rms)
7. [Paso 5 — Trim de silencio](#7-paso-5--trim-de-silencio)
8. [Paso 6 — Extracción del embedding (HuBERT)](#8-paso-6--extracción-del-embedding-hubert)
9. [Paso 7 — Clasificación de emoción](#9-paso-7--clasificación-de-emoción)
10. [Particularidades según el método de entrada](#10-particularidades-según-el-método-de-entrada)
11. [Referencia rápida de parámetros](#11-referencia-rápida-de-parámetros)

---

## 1. Entradas soportadas

El módulo web acepta audio mediante dos métodos:

| Método | Descripción | Consideraciones especiales |
|---|---|---|
| **Carga de archivo** | El usuario sube un `.wav` desde su equipo | Puede venir con cualquier sample rate, canales o nivel de volumen |
| **Grabación en vivo** | Captura directa desde el micrófono del navegador (`MediaRecorder API`) | El navegador entrega el audio en el formato nativo del codec; debe convertirse a `.wav`/PCM antes de procesar |

Independientemente del método de entrada, **ambos flujos convergen en el mismo
pipeline de preprocesamiento** desde el Paso 1 en adelante. No existe un pipeline
diferente para grabación en vivo y para archivo.

> Decidir factibilidad de cada uno y tomar una desicion

---

## 2. Resumen ejecutivo del pipeline

```
[ Audio crudo del usuario ]
           │
           ▼
   ┌────────────────┐
   │ Paso 1         │  Decodificar a PCM float32
   └───────┬────────┘
           ▼
   ┌────────────────┐
   │ Paso 2         │  Convertir a mono (si es estéreo)
   └───────┬────────┘
           ▼
   ┌────────────────┐
   │ Paso 3         │  Resamplear a 16 000 Hz (si no lo es ya)
   └───────┬────────┘
           ▼
   ┌────────────────┐
   │ Paso 4         │  Normalización de RMS (ajuste de volumen)
   └───────┬────────┘
           ▼
   ┌────────────────┐
   │ Paso 5         │  Trim de silencio de inicio y fin
   └───────┬────────┘
           ▼
   ┌────────────────┐
   │ Paso 6         │  Forward pass por HuBERT + mean pooling → vector [768]
   └───────┬────────┘
           ▼
   ┌────────────────┐
   │ Paso 7         │  Cabeza clasificadora → softmax → 7 probabilidades
   └───────┬────────┘
           ▼
   [ Emoción predicha + distribución de probabilidades ]
```

El **audio nunca se segmenta en ventanas temporales**: el clip completo se
procesa de una sola vez. Los clips de entrenamiento (RAVDESS) son expresiones
emocionales completas de ~3–5 s; segmentarlos degradaría la calidad de la
predicción. Esta decisión aplica también a la grabación en vivo: el modelo
recibe el clip completo de la locución, no fragmentos.

---

## 3. Paso 1 — Decodificación del audio

### Qué hace
Leer el archivo de audio desde disco (o desde el buffer en memoria si viene de
la grabación en vivo) y obtener los datos de señal como un array de punto
flotante de 32 bits (`float32`).

### Cómo implementarlo

```python
import soundfile as sf
import numpy as np

data, sr = sf.read(ruta_archivo, dtype="float32")
# data: array numpy shape (N,) si es mono, (N, canales) si es estéreo
# sr:   sample rate original del archivo
```

Para grabaciones del navegador en memoria (bytes recibidos por el servidor):

```python
import io
import soundfile as sf

buffer = io.BytesIO(bytes_recibidos_del_navegador)
data, sr = sf.read(buffer, dtype="float32")
```

### Por qué float32
El modelo HuBERT espera un tensor `float32`. Convertir en la lectura evita una
conversión posterior y no introduce pérdida de precisión relevante para audio.

---

## 4. Paso 2 — Conversión a mono

### Qué hace
Si el audio tiene más de un canal (estéreo, surround, etc.), reducirlo a un
único canal antes de continuar.

### Por qué
HuBERT fue entrenado sobre señales mono. El dataset de entrenamiento (RAVDESS)
incluye archivos en estéreo que fueron convertidos a mono en la etapa de
preparación de datos. Si el audio de inferencia llega en estéreo y no se
convierte, las dimensiones del tensor no coincidirán con lo que el modelo
espera.

### Cómo implementarlo

```python
if data.ndim > 1:
    # Promedio aritmético de los canales → mono
    data = data.mean(axis=1).astype(np.float32)
# A partir de aquí: data.shape == (N,)  —  array 1D
```

**Nota:** el promedio de canales es la misma operación aplicada durante el
entrenamiento. No usar `data[:, 0]` (tomar solo el canal izquierdo), porque
introduce asimetría respecto al proceso de entrenamiento.

---

## 5. Paso 3 — Resampling a 16 000 Hz

### Qué hace
Si el audio no está ya a 16 000 Hz, convertir su tasa de muestreo a ese valor.

### Por qué
HuBERT (`facebook/hubert-base-ls960`) fue pre-entrenado con audio a **16 kHz**.
Pasarle audio a otra frecuencia provoca que el modelo interprete el tempo del
habla de forma incorrecta y produce embeddings degradados. El dataset RAVDESS
original está a 48 kHz; durante el entrenamiento se hizo resampling explícito a
16 kHz antes de procesar cualquier archivo.

Los archivos grabados desde un navegador suelen llegar a 44 100 Hz o 48 000 Hz
dependiendo del hardware del usuario, por lo que este paso es obligatorio para
el flujo de grabación en vivo.

### Cómo implementarlo

```python
import librosa   # alternativa: scipy.signal.resample o resampy

SAMPLE_RATE_TARGET = 16_000

if sr != SAMPLE_RATE_TARGET:
    data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE_TARGET)
    sr = SAMPLE_RATE_TARGET
```

**Nota sobre la librería de resampling:** `librosa.resample` usa por defecto
el algoritmo de Kaiser (alta calidad). Cualquier librería de resampling de
calidad similar es aceptable, pero no usar técnicas de baja calidad como
decimación simple sin filtro antialiasing.

---

## 6. Paso 4 — Normalización de RMS

### Qué hace
Ajustar el nivel de volumen del audio para que su energía cuadrática media
(RMS) alcance un valor de referencia fijo.

### Por qué
En el dataset de entrenamiento se observaron diferencias grandes de nivel entre
los clips de distintos actores. La normalización de RMS hace que todos los
archivos lleguen al modelo con un nivel de energía comparable, eliminando el
volumen como variable espuria que el modelo podría aprender a explotar.
Para que la distribución de la señal en inferencia sea estadísticamente
consistente con la de entrenamiento, **se debe aplicar la misma normalización**.

### Cómo implementarlo

```python
TARGET_RMS = 0.1   # valor de referencia usado durante el entrenamiento

def normalizar_rms(data: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    rms_actual = np.sqrt(np.mean(data ** 2))
    if rms_actual < 1e-8:
        # Clip casi en silencio: no escalar para evitar amplificar ruido de fondo
        return data
    return (data * (target_rms / rms_actual)).astype(np.float32)

data = normalizar_rms(data)
```

> **Aviso — decisión técnica pendiente de confirmación.**  
> El feature extractor estándar de HuggingFace para modelos de la familia
> HuBERT/wav2vec2 incluye una normalización interna por utterance (media 0,
> varianza 1). El código de entrenamiento actual **no usa** ese feature extractor:
> alimenta directamente el waveform preprocesado a `HubertModel`, por lo que esa
> normalización interna no se aplica. Esta es la razón por la que la normalización
> de RMS manual del Paso 4 **sí es relevante para inferencia** y debe mantenerse.
> Si en el futuro se decide incorporar el feature extractor de HuggingFace con
> `do_normalize=True`, este Paso 4 debe revisarse para evitar una doble
> normalización.

---

## 7. Paso 5 — Trim de silencio

### Qué hace
Recortar los silencios del **inicio y del final** del clip. Los silencios
internos (pausas naturales entre palabras) no se tocan.

### Por qué
Los clips de RAVDESS incluyen márgenes de silencio antes y después de la
locución. Recortarlos hace que el contenido emocional ocupe una mayor proporción
del clip, mejorando la calidad del embedding. En inferencia, un audio grabado
desde el navegador puede incluir silencio de rampa antes de que el usuario
empiece a hablar; este paso lo elimina automáticamente.

### Algoritmo (replicado del entrenamiento)

La detección es **relativa al pico del propio clip**: se mide la energía RMS
en ventanas solapadas y se descartan, por los extremos, las ventanas que estén
más de `TRIM_TOP_DB` dB por debajo del frame más fuerte.

**Parámetros exactos usados en entrenamiento:**

| Parámetro | Valor | Descripción |
|---|---|---|
| `FRAME_LENGTH` | `2048` muestras | Tamaño de la ventana para medir energía |
| `HOP_LENGTH` | `512` muestras | Salto entre ventanas consecutivas |
| `TRIM_TOP_DB` | `30.0` dB | Umbral: frames con energía < pico − 30 dB se consideran silencio |

### Cómo implementarlo

```python
FRAME_LENGTH = 2048
HOP_LENGTH   = 512
TRIM_TOP_DB  = 30.0

def _rms_por_frame(data, frame_length, hop_length):
    if len(data) < frame_length:
        return np.array([np.sqrt(np.mean(data ** 2))]) if len(data) else np.array([0.0])
    n_frames = 1 + (len(data) - frame_length) // hop_length
    rms = np.empty(n_frames)
    for i in range(n_frames):
        inicio = i * hop_length
        rms[i] = np.sqrt(np.mean(data[inicio:inicio + frame_length] ** 2))
    return rms

def detectar_limites(data, frame_length, hop_length, top_db):
    if len(data) == 0:
        return 0, 0
    rms = _rms_por_frame(data, frame_length, hop_length)
    ref = rms.max()
    if ref <= 0:
        return 0, len(data)
    db = 20 * np.log10(np.maximum(rms, 1e-10) / ref)
    no_silencio = np.nonzero(db > -top_db)[0]
    if len(no_silencio) == 0:
        return 0, len(data)
    inicio_frame, fin_frame = no_silencio[0], no_silencio[-1]
    inicio_muestra = inicio_frame * hop_length
    fin_muestra = min(len(data), fin_frame * hop_length + frame_length)
    return int(inicio_muestra), int(fin_muestra)

inicio, fin = detectar_limites(data, FRAME_LENGTH, HOP_LENGTH, TRIM_TOP_DB)
data = data[inicio:fin]
```

**Caso borde:** si el clip queda con menos de `0.5 s` de duración tras el trim,
es una señal de que el audio de entrada era casi todo silencio o era demasiado
corto. En este caso, devolver un error al usuario antes de continuar.

```python
MIN_DURACION_S = 0.5

if len(data) / SAMPLE_RATE_TARGET < MIN_DURACION_S:
    raise ValueError(
        "El audio es demasiado corto o contiene casi solo silencio "
        f"(duración tras trim: {len(data) / SAMPLE_RATE_TARGET:.2f} s). "
        "Por favor, suba o grabe un clip de al menos 0.5 s con contenido de voz."
    )
```

---

## 8. Paso 6 — Extracción del embedding (HuBERT)

### Qué hace
Pasar el waveform preprocesado por el modelo HuBERT congelado y obtener un
único vector de 768 dimensiones que representa el contenido emocional del clip.

### Componentes

| Componente | Detalle |
|---|---|
| **Modelo** | `facebook/hubert-base-ls960` (HuBERT base, LibriSpeech 960h) |
| **Estado** | Completamente congelado (`requires_grad=False`, `model.eval()`) |
| **Librería** | `transformers.HubertModel` de HuggingFace |
| **Pooling** | Mean pooling sobre la dimensión temporal de la última capa oculta |
| **Salida** | Vector `float32` de dimensión `768` |

### Cómo implementarlo

```python
import torch
from transformers import HubertModel

# Cargar y congelar HuBERT (hacerlo UNA sola vez al iniciar el servidor)
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
for p in hubert.parameters():
    p.requires_grad = False
hubert.eval()
hubert.to(dispositivo)

@torch.no_grad()
def extraer_embedding(waveform: np.ndarray) -> np.ndarray:
    """waveform: array float32 1D a 16 kHz, ya preprocesado.
       Devuelve: array float32 shape (768,).
    """
    entrada = torch.from_numpy(waveform).unsqueeze(0).to(dispositivo)  # [1, T]
    salida  = hubert(entrada).last_hidden_state                         # [1, F, 768]
    emb     = salida.mean(dim=1).squeeze(0)                             # [768]
    return emb.cpu().numpy().astype(np.float32)
```

**Puntos críticos:**

- HuBERT se carga una sola vez cuando el servidor arranca, **no en cada petición**.
  Cargarlo por petición es inviable en producción (el modelo pesa ~360 MB y tarda
  varios segundos en inicializar).
- El `@torch.no_grad()` es obligatorio para inferencia: sin él, PyTorch acumula
  gradientes innecesarios, multiplicando el uso de memoria.
- El tensor de entrada tiene shape `[1, T]` (batch de 1, T muestras). No hay
  padding ni truncado: el clip completo se procesa en un solo forward pass.

---

## 9. Paso 7 — Clasificación de emoción

### Qué hace
Tomar el vector de 768 dimensiones y producir una distribución de probabilidad
sobre las 7 emociones de la taxonomía unificada.

### Arquitectura de la cabeza clasificadora

```
entrada [768]
    │
    ▼
Dropout(p=0.3)
    │
    ▼
Linear(768 → 256) + ReLU
    │
    ▼
Dropout(p=0.3)
    │
    ▼
Linear(256 → 7)   ← logits
    │
    ▼
Softmax(dim=1)    ← probabilidades
```

| Parámetro | Valor |
|---|---|
| `dim_entrada` | 768 |
| `dim_oculta` | 256 |
| `dropout` | 0.3 |
| `num_clases` | 7 |

### Taxonomía de salida (orden fijo)

El índice de clase es relevante porque el checkpoint guarda los pesos ordenados
según esta tabla. **No alterar el orden.**

| Índice | Etiqueta interna | Nombre en español |
|---|---|---|
| 0 | `neutral` | Neutral |
| 1 | `joy` | Felicidad |
| 2 | `sadness` | Tristeza |
| 3 | `anger` | Enojo |
| 4 | `fear` | Miedo |
| 5 | `disgust` | Disgusto |
| 6 | `surprise` | Sorpresa |

### Cómo implementarlo

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def clasificar(embedding: np.ndarray, cabeza, dispositivo) -> dict:
    """embedding: array float32 shape (768,).
       Devuelve dict con emoción predicha y ranking completo.
    """
    EMOCIONES = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
    NOMBRES_ES = {
        "neutral": "neutral", "joy": "felicidad", "sadness": "tristeza",
        "anger": "enojo",     "fear": "miedo",    "disgust": "disgusto",
        "surprise": "sorpresa",
    }

    x     = torch.from_numpy(embedding).unsqueeze(0).to(dispositivo)  # [1, 768]
    probs = F.softmax(cabeza(x), dim=1).squeeze(0).cpu().numpy()       # [7]

    ranking = sorted(
        [(EMOCIONES[i], float(probs[i])) for i in range(len(EMOCIONES))],
        key=lambda t: t[1], reverse=True,
    )
    emocion, confianza = ranking[0]
    return {
        "emocion":    emocion,
        "emocion_es": NOMBRES_ES[emocion],
        "confianza":  confianza,
        "ranking":    ranking,         # lista de (emocion, prob) de mayor a menor
    }
```

---

## 10. Particularidades según el método de entrada

### 10.1 Archivo `.wav` cargado por el usuario

El pipeline se aplica en orden completo desde el Paso 1. No se asume nada sobre
el archivo: puede venir en mono o estéreo, a cualquier frecuencia de muestreo y
con cualquier nivel de volumen.

Validaciones recomendadas **antes** de ejecutar el pipeline (para dar feedback
temprano al usuario):

- Verificar que el archivo tiene extensión `.wav` y es un PCM válido (intentar
  decodificar; capturar error si falla).
- Rechazar archivos excesivamente largos (> 30 s) antes de procesarlos: el
  modelo no fue entrenado con clips de esa duración y el resultado sería
  incierto. Mostrar un mensaje explicativo al usuario.

### 10.2 Grabación en vivo desde el navegador

La `MediaRecorder API` del navegador entrega el audio en un formato que depende
del navegador y el sistema operativo (típicamente `audio/webm;codecs=opus` en
Chrome/Firefox). **Este formato no es WAV/PCM**, por lo que se requiere un paso
de conversión adicional **antes** del Paso 1.

Opciones:

**Opción A (recomendada) — conversión en el backend:**  
Enviar los bytes del audio tal como los entrega `MediaRecorder` al servidor y
decodificar con `soundfile` o con `ffmpeg` vía `pydub`. `soundfile` soporta
WebM/Opus en algunas instalaciones; `pydub` vía `ffmpeg` es más portable.

```python
from pydub import AudioSegment
import io, numpy as np

audio_seg = AudioSegment.from_file(io.BytesIO(bytes_recibidos), format="webm")
data = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
data /= 2 ** (audio_seg.sample_width * 8 - 1)   # normalizar int → float [-1, 1]
sr   = audio_seg.frame_rate
# A partir de aquí: continuar desde el Paso 2 con `data` y `sr`
```

**Opción B — conversión en el frontend:**  
Usar la `Web Audio API` para decodificar el stream grabado y reensamplarlo a
16 kHz antes de enviar al servidor. Reduce la carga del backend pero agrega
complejidad en el cliente. Solo tiene sentido si el servidor tiene limitaciones
de cómputo muy estrictas.

En ambos casos, **los pasos 2 al 7 se aplican igual** una vez que el audio está
en formato PCM float32.

---

## 11. Referencia rápida de parámetros

Todos los valores listados aquí son los que se usaron durante el entrenamiento.
No cambiarlos sin revisar el impacto en la distribución de entrada al modelo.

| Parámetro | Valor | Etapa |
|---|---|---|
| `SAMPLE_RATE_TARGET` | `16 000 Hz` | Pasos 3 y 6 |
| `TARGET_RMS` | `0.1` | Paso 4 |
| `FRAME_LENGTH` | `2048 muestras` | Paso 5 |
| `HOP_LENGTH` | `512 muestras` | Paso 5 |
| `TRIM_TOP_DB` | `30.0 dB` | Paso 5 |
| `MIN_DURACION_S` | `0.5 s` | Paso 5 (validación) |
| `HUBERT_MODEL_NAME` | `facebook/hubert-base-ls960` | Paso 6 |
| `EMBEDDING_DIM` | `768` | Paso 6 |
| `HIDDEN_DIM` | `256` | Paso 7 |
| `DROPOUT` | `0.3` | Paso 7 |
| `NUM_CLASSES` | `7` | Paso 7 |

---

*Documento generado el 2026-06-23 a partir del código fuente de `src/` y del
contexto de entrenamiento documentado en `CLAUDE.md`.*
