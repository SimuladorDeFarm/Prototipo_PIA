# Guía de los Archivos de Inferencia — Módulo de Voz
## Para quien construye el nuevo programa

> Este documento explica qué hace cada archivo de `src/` que fue copiado al
> nuevo proyecto. El objetivo es que puedas entender el código existente y,
> si lo necesitas, reescribir partes de él en otro lenguaje o con otra
> estructura, sin romper la compatibilidad con el modelo entrenado.

---

## Mapa de dependencias

```
inferencia.py          ← punto de entrada principal
    ├── modelo.py      ← carga HuBERT (congelado)
    ├── embeddings.py  ← carga audio + extrae embedding
    ├── clasificador.py← define la arquitectura de la red (CRÍTICO)
    └── config.py      ← constantes compartidas por todos
```

El flujo de datos es siempre el mismo:

```
audio .wav
    → embeddings.py (cargar_audio)
    → embeddings.py (extraer_embedding con HuBERT de modelo.py)
    → clasificador.py (CabezaEmocion.forward)
    → softmax → emoción + probabilidades
```

---

## `config.py` — Constantes del sistema

No tiene lógica. Es un archivo de configuración que define todos los valores
numéricos y rutas que usan los otros módulos. Si no lo usas, simplemente
reemplaza cada referencia a `config.X` por su valor literal.

**Los valores que importan para inferencia:**

```python
SAMPLE_RATE_TARGET = 16_000        # Hz requeridos por HuBERT
HUBERT_MODEL_NAME  = "facebook/hubert-base-ls960"
EMBEDDING_DIM      = 768           # dimensión del vector que produce HuBERT
HIDDEN_DIM         = 256           # neuronas de la capa oculta de la cabeza
DROPOUT            = 0.3
EMOCIONES_7        = ["neutral", "joy", "sadness", "anger",
                       "fear", "disgust", "surprise"]
```

Los demás valores (`DATASET_*`, `EPOCHS`, `SEED`, etc.) son para
entrenamiento. En inferencia no se usan.

---

## `clasificador.py` — La red neuronal (ARCHIVO CRÍTICO)

Define `CabezaEmocion`, la única red que **se entrenó y cuyos pesos están
guardados en `clasificador_voz.pt`**. Es el único archivo que no puedes
reimplementar libremente: la arquitectura tiene que ser idéntica a esta para
que `load_state_dict()` cargue los pesos correctamente.

```python
class CabezaEmocion(nn.Module):
    def __init__(self, dim_entrada=768, dim_oculta=256, num_clases=7, dropout=0.3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_entrada, dim_oculta),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_oculta, num_clases),
        )

    def forward(self, x):
        return self.red(x)   # devuelve logits [batch, 7], NO probabilidades
```

**Puntos clave:**

- Recibe un tensor `[batch, 768]` y devuelve **logits** `[batch, 7]`, no
  probabilidades. Para obtener probabilidades hay que aplicar `softmax` después.
- El `Dropout` está presente en la definición porque forma parte del grafo que
  se guarda. En inferencia, al poner el modelo en `.eval()`, PyTorch
  automáticamente desactiva el dropout — no hay que quitarlo del código.
- El orden de los pesos internos (`red.0`, `red.1`, `red.2`, `red.3`, `red.4`)
  corresponde al orden de capas del `nn.Sequential`. Si cambias el orden o los
  nombres, `load_state_dict()` fallará.

---

## `modelo.py` — Carga de HuBERT

Wrapper sobre HuggingFace `transformers`. Su única función real es descargar
HuBERT, congelar sus parámetros y ponerlo en modo evaluación.

**La función que usarás:**

```python
def cargar_hubert(nombre=None, device=None, verbose=True):
    # nombre  → "facebook/hubert-base-ls960" por defecto
    # device  → "cuda", "cpu", o None (autodetecta GPU si hay)
    # verbose → imprime progreso si True, silencioso si False
    return modelo, dispositivo
```

**Lo que hace internamente (puedes replicarlo en 4 líneas):**

```python
from transformers import HubertModel
import torch

modelo = HubertModel.from_pretrained("facebook/hubert-base-ls960")
for p in modelo.parameters():
    p.requires_grad = False
modelo.eval()
modelo.to(dispositivo)
```

La primera llamada descarga ~360 MB desde HuggingFace y los guarda en caché
local (`~/.cache/huggingface/`). Las llamadas siguientes cargan desde caché
en segundos.

**Importante:** HuBERT debe cargarse **una sola vez** cuando arranca el
servidor, no en cada petición. Cargarlo por petición haría el sistema inutilizable.

---

## `embeddings.py` — Carga de audio y extracción del embedding

Tiene dos funciones relevantes para inferencia. El resto del archivo
(`extraer_embeddings`, `_leer_indice`, `_imprimir_resumen`) es para procesar
el dataset completo en batch — no se usa en inferencia.

### `cargar_audio(ruta)`

```python
def cargar_audio(ruta):
    data, sr = sf.read(ruta, dtype="float32")
    if data.ndim > 1:                    # si es estéreo: promediar canales
        data = data.mean(axis=1).astype(np.float32)
    if sr != 16_000:
        raise ValueError(f"sample rate {sr}, se esperaba 16000")
    return data    # array float32 1D, a 16 kHz
```

Carga el `.wav` como `float32` y verifica que ya esté a 16 kHz. **No hace
resampling**: asume que el audio ya fue preprocesado. Para inferencia web
debes aplicar el pipeline de `pipeline_audio.md` antes de llamar a esta
función (o reemplazarla por tu propia carga que sí incluya el resampling).

### `extraer_embedding(modelo, dispositivo, waveform)`

```python
@torch.no_grad()
def extraer_embedding(modelo, dispositivo, waveform):
    entrada = torch.from_numpy(waveform).unsqueeze(0).to(dispositivo)  # [1, T]
    salida  = modelo(entrada).last_hidden_state                         # [1, F, 768]
    embedding = salida.mean(dim=1).squeeze(0)                          # [768]
    return embedding.cpu().numpy().astype(np.float32)
```

- `waveform` es el array `float32` 1D devuelto por `cargar_audio`.
- Hace un forward pass por HuBERT sin calcular gradientes (`@torch.no_grad()`).
- `last_hidden_state` es la salida de la última capa de transformers: shape
  `[1, F, 768]` donde `F` es el número de frames internos de HuBERT (varía
  según la duración del audio).
- **Mean pooling**: `mean(dim=1)` promedia todos los frames en un único vector
  `[768]`. Esta es la decisión de pooling usada en entrenamiento — si cambias
  este paso (ej. max pooling o usar solo el primer frame), el embedding
  resultante será diferente y el clasificador producirá predicciones incorrectas.
- Devuelve un array numpy `float32` de shape `(768,)`.

---

## `inferencia.py` — Punto de entrada (conveniencia)

Reúne los tres módulos anteriores en una interfaz de alto nivel. No tiene
lógica propia relevante — todo lo que hace es llamar a los otros módulos.

### `cargar_clasificador(ruta_checkpoint, dispositivo)`

Carga el `.pt` y reconstruye `CabezaEmocion` con los parámetros guardados.

```python
estado = torch.load(ruta, map_location=dispositivo, weights_only=False)
# estado es un dict con:
#   "model_state"   → los pesos de la red (para load_state_dict)
#   "embedding_dim" → 768
#   "hidden_dim"    → 256
#   "num_classes"   → 7  (implícito en el tamaño de la capa de salida)
#   "dropout"       → 0.3
#   "emociones"     → ["neutral", "joy", "sadness", ...]
#   "metadata"      → dict con fecha, métricas, etc.

modelo = CabezaEmocion(
    dim_entrada=estado["embedding_dim"],
    dim_oculta=estado["hidden_dim"],
    num_clases=len(estado["emociones"]),
    dropout=estado["dropout"],
).to(dispositivo)
modelo.load_state_dict(estado["model_state"])
modelo.eval()
```

El checkpoint es **autosuficiente**: contiene la arquitectura y los pesos.
No necesitas saber de memoria que `dim_oculta=256`; lo lees del propio `.pt`.

### `class Predictor`

Wrapper de conveniencia que carga HuBERT y la cabeza una vez y expone
`predecir(ruta_audio)`.

```python
predictor = Predictor(ruta_checkpoint="modelos/clasificador_voz.pt")
resultado  = predictor.predecir("audio_preprocesado.wav")
```

`resultado` es un dict:

```python
{
    "emocion":    "joy",           # etiqueta interna en inglés
    "emocion_es": "felicidad",     # nombre en español
    "confianza":  0.87,            # probabilidad de la clase ganadora
    "ranking": [                   # todas las clases, de mayor a menor prob
        ("joy",      0.87),
        ("neutral",  0.06),
        ("sadness",  0.04),
        ...
    ]
}
```

**Limitación importante:** `predecir()` llama a `cargar_audio()`, que asume
que el audio ya está a 16 kHz y no hace resampling. Para inferencia web,
necesitas preprocesar primero y luego llamar directamente a `extraer_embedding`
+ `clasificar`, o modificar `predecir()` para que acepte un waveform ya
procesado en lugar de una ruta.

---

## Cómo construir la inferencia sin usar `Predictor`

Si prefieres escribir el flujo de cero en tu nuevo programa, estos son los
pasos mínimos con el código equivalente:

```python
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import HubertModel
# Solo necesitas copiar CabezaEmocion de clasificador.py

EMOCIONES = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Cargar modelos (una sola vez al arrancar) ---
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
for p in hubert.parameters():
    p.requires_grad = False
hubert.eval().to(dispositivo)

estado = torch.load("modelos/clasificador_voz.pt",
                    map_location=dispositivo, weights_only=False)
cabeza = CabezaEmocion(
    dim_entrada=estado["embedding_dim"],
    dim_oculta=estado["hidden_dim"],
    num_clases=len(estado["emociones"]),
    dropout=estado["dropout"],
).to(dispositivo)
cabeza.load_state_dict(estado["model_state"])
cabeza.eval()

# --- 2. Por cada petición: preprocesar audio y predecir ---
# (el audio ya debe venir preprocesado: mono, 16 kHz, RMS norm, trim)
data, sr = sf.read("audio.wav", dtype="float32")

with torch.no_grad():
    entrada   = torch.from_numpy(data).unsqueeze(0).to(dispositivo)  # [1, T]
    emb       = hubert(entrada).last_hidden_state.mean(dim=1)        # [1, 768]
    probs     = F.softmax(cabeza(emb), dim=1).squeeze(0).cpu().numpy()

idx_ganador = probs.argmax()
print(f"Emoción: {EMOCIONES[idx_ganador]}, confianza: {probs[idx_ganador]:.2%}")
```

---

## Resumen: qué puedes cambiar y qué no

| Elemento | ¿Puedes cambiarlo? | Por qué |
|---|---|---|
| Arquitectura de `CabezaEmocion` | **No** | Los pesos del `.pt` son para esta arquitectura exacta |
| Cómo cargas HuBERT | Sí | Mientras uses el mismo checkpoint `facebook/hubert-base-ls960` |
| La operación de pooling (`mean`) | **No** | El clasificador fue entrenado con embeddings de mean pooling |
| Cómo lees el `.wav` | Sí | Mientras el resultado sea `float32`, mono, 16 kHz |
| Las constantes de `config.py` | Sí | Son valores, no pesos entrenados |
| La clase `Predictor` | Sí | Es solo conveniencia |

*Documento generado el 2026-06-23.*
