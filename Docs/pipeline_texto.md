# Instructivo de Preprocesamiento de Texto para Inferencia

**Módulo:** Texto — NeuroEmoInnovat (PIA)
**Modelo:** BETO (`dccuchile/bert-base-spanish-wwm-cased`) fine-tuneado sobre EMOEvent
**Propósito:** Este documento define el pipeline de transformaciones que debe aplicarse a cualquier texto de entrada **antes** de pasarlo al modelo en modo inferencia. Cada paso replica exactamente el preprocesamiento usado durante el entrenamiento.

---

## Contexto general

El modelo fue entrenado con textos ya limpios del corpus EMOEvent (tweets en español). Para que las predicciones sean coherentes con lo que el modelo aprendió, el texto de entrada debe pasar por las mismas transformaciones que se aplicaron durante el entrenamiento, en el mismo orden.

**Restricción crítica:** el modelo usa la variante **cased** de BETO. Esto significa que es sensible a las mayúsculas. **No se debe convertir el texto a minúsculas** en ningún paso del pipeline.

---

## Pipeline de preprocesamiento (orden obligatorio)

```
Texto crudo
    │
    ▼
[Paso 1] Conversión de emojis
    │
    ▼
[Paso 2] Eliminación de ruido (URLs, placeholders, símbolo #)
    │
    ▼
[Paso 3] Tokenización con BETO (WordPiece)
    │
    ▼
Tensores listos para el modelo
```

---

## Paso 1 — Conversión de emojis a texto en español

### Qué hace

Cada emoji se reemplaza por su descripción en español rodeada de espacios. Los emojis consecutivos quedan separados por espacios. Los caracteres `:` y `_` que ya existían en el texto original (p. ej. URLs, horarios como `10:30`) **no se modifican**.

### Por qué este orden

El paso de emojis debe ir **antes** de la eliminación de URLs y placeholders: si un tweet tiene una URL seguida de un emoji (`https://ejemplo.com😭`), la eliminación de la URL en el paso 2 deja el emoji intacto para que el paso 1 lo convierta. Si se hiciera al revés, podría quedar un emoji sin procesar.

### Implementación de referencia

```python
import re
import emoji

_DELIM = ("\x00", "\x01")
_TOKEN_EMOJI = re.compile("\x00([^\x01]*)\x01")

def convertir_emojis(texto: str) -> str:
    demoj = emoji.demojize(texto, language="es", delimiters=_DELIM)
    convertido = _TOKEN_EMOJI.sub(
        lambda m: " " + m.group(1).replace("_", " ").strip() + " ", demoj
    )
    return emoji.replace_emoji(convertido, "")
```

**Dependencia:** librería `emoji` (`pip install emoji`).

### Ejemplo

| Entrada | Salida |
|---|---|
| `"Hola😭 no puedo más"` | `"Hola cara llorando fuerte  no puedo más"` |
| `"genial!!🎉🎉"` | `"genial!! fiesta con serpentinas   fiesta con serpentinas "` |

---

## Paso 2 — Eliminación de ruido (URLs, placeholders y símbolo `#`)

### Qué hace

Elimina los siguientes elementos del texto:

| Elemento | Patrón | Acción |
|---|---|---|
| URLs reales | `http://...` y `https://...` | Se elimina la URL completa |
| Placeholders del corpus | Las palabras `HASHTAG`, `URL`, `USER` (en mayúscula, como palabra completa) | Se eliminan |
| Símbolo de hashtag | `#` | Se elimina el símbolo **pero se conserva la palabra** |
| Espacios múltiples | Dos o más espacios consecutivos | Se colapsan en uno solo |

**Nota sobre los placeholders:** EMOEvent reemplazó los `#`, `@` y URLs originales por las palabras `HASHTAG`, `USER`, `URL` en mayúscula. El corpus de entrenamiento fue limpiado eliminando esas palabras; en inferencia con texto libre esas palabras no aparecerán, pero el paso sigue siendo necesario para eliminar URLs reales y el símbolo `#`.

### Implementación de referencia

```python
import re

_PATRON_PLACEHOLDERS = re.compile(r"\b(?:HASHTAG|URL|USER)\b")
_PATRON_URL = re.compile(r"http\S+")

def eliminar_placeholders(texto: str) -> str:
    sin = _PATRON_URL.sub(" ", texto)
    sin = _PATRON_PLACEHOLDERS.sub(" ", sin)
    sin = sin.replace("#", "")
    return re.sub(r"\s{2,}", " ", sin).strip()
```

### Ejemplo

| Entrada | Salida |
|---|---|
| `"mira esto https://t.co/abc #Educación"` | `"mira esto  Educación"` |
| `"gran día HASHTAG felicidad"` | `"gran día  felicidad"` |

---

## Paso 3 — Tokenización con BETO (WordPiece)

### Qué hace

El texto limpio se tokeniza usando el tokenizador nativo de BETO (`AutoTokenizer` de HuggingFace). La tokenización produce tres tensores que son los que recibe el modelo:

| Tensor | Descripción | Valores |
|---|---|---|
| `input_ids` | IDs de cada token en el vocabulario de BETO | Enteros en `[0, vocab_size)` |
| `attention_mask` | Indica qué posiciones son tokens reales (1) y cuáles son padding (0) | `0` o `1` |
| `token_type_ids` | Segmento al que pertenece cada token | Todo `0` (clasificación de secuencia única) |

### Parámetros (deben ser idénticos a los usados en entrenamiento)

| Parámetro | Valor | Razón |
|---|---|---|
| `model_name` | `"dccuchile/bert-base-spanish-wwm-cased"` | Variante cased, en español |
| `max_length` | `128` | Longitud fija usada en entrenamiento |
| `padding` | `"max_length"` | Rellena hasta 128 tokens con `[PAD]` |
| `truncation` | `True` | Corta si supera 128 tokens |
| `return_token_type_ids` | `True` | Requerido por la arquitectura BERT |

### Implementación de referencia

```python
from transformers import AutoTokenizer

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128

def cargar_tokenizador():
    return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

def tokenizar(texto: str, tokenizer):
    return tokenizer(
        texto,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
```

**Resultado:** un diccionario con tres tensores de forma `[1, 128]` (batch de un solo ejemplo).

### Notas sobre la longitud

- La longitud fija de 128 cubre el 100% del corpus EMOEvent sin truncar.
- En inferencia, textos más largos se truncarán. Textos más cortos se rellenarán con padding.
- El token especial `[CLS]` ocupa la posición 0; `[SEP]` cierra la secuencia. Ambos los inserta el tokenizador automáticamente.

---

## Función completa de preprocesamiento para inferencia

Esta función integra los tres pasos en el orden correcto y devuelve los tensores listos para pasarlos al modelo:

```python
def preprocesar_para_inferencia(texto: str, tokenizer) -> dict:
    texto_limpio = convertir_emojis(texto)
    texto_limpio = eliminar_placeholders(texto_limpio)
    return tokenizar(texto_limpio, tokenizer)
```

**Uso en inferencia:**

```python
import torch

tokenizer = cargar_tokenizador()
modelo.eval()

with torch.no_grad():
    inputs = preprocesar_para_inferencia(texto_usuario, tokenizer)
    logits = modelo(**inputs).logits
    id_predicho = logits.argmax(dim=-1).item()
```

---

## Paso 4 — Interpretación de la salida del modelo

El modelo devuelve 7 logits. El índice con el valor más alto es la emoción predicha. El mapeo es:

| ID | Etiqueta del modelo | Emoción (taxonomía unificada) |
|---|---|---|
| `0` | `others` | Neutral |
| `1` | `joy` | Felicidad |
| `2` | `sadness` | Tristeza |
| `3` | `anger` | Enojo |
| `4` | `fear` | Miedo |
| `5` | `disgust` | Disgusto |
| `6` | `surprise` | Sorpresa |

```python
ID2LABEL = {
    0: "others",
    1: "joy",
    2: "sadness",
    3: "anger",
    4: "fear",
    5: "disgust",
    6: "surprise",
}

id_predicho = logits.argmax(dim=-1).item()
emocion = ID2LABEL[id_predicho]
```

Si se necesita la probabilidad de cada clase (p. ej. para mostrar una barra de confianza):

```python
import torch.nn.functional as F

probabilidades = F.softmax(logits, dim=-1).squeeze().tolist()
```

---

## Carga del checkpoint entrenado

El checkpoint v4 se encuentra en `backend/models/texto/clasificador_texto_v4.pt`. Para cargarlo en modo inferencia:

```python
import torch
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
NUM_LABELS = 7
LABEL2ID = {"others": 0, "joy": 1, "sadness": 2, "anger": 3,
            "fear": 4, "disgust": 5, "surprise": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def cargar_modelo_inferencia(ruta_checkpoint, device="cpu"):
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    state_dict = torch.load(ruta_checkpoint, map_location=device, weights_only=True)
    modelo.load_state_dict(state_dict)
    modelo.to(device)
    modelo.eval()
    return modelo
```

---

## Resumen de lo que NO se hace en inferencia

Estas operaciones ocurren solo durante el entrenamiento y **no deben replicarse en inferencia**:

| Operación | Por qué no aplica en inferencia |
|---|---|
| Cálculo de class weights | Son pesos de la función de pérdida (loss), que no existe en inferencia |
| Label encoding (emoción → entero) | En inferencia se parte del texto, no de etiquetas |
| DataLoader / batching automático | El prototipo procesa un texto a la vez; el batch es de tamaño 1 |
| Shuffle de datos | No hay un dataset de entrenamiento que ordenar |
| Gradientes (`loss.backward()`) | El modelo está en `eval()` y dentro de `torch.no_grad()` |

---

## Checklist de verificación antes de conectar al modelo

- [ ] El texto de entrada **no fue convertido a minúsculas** en ningún paso.
- [ ] Los emojis fueron convertidos a texto en español **antes** de eliminar las URLs.
- [ ] Los tensores de salida tienen forma `[1, 128]` (`batch_size=1`, `max_length=128`).
- [ ] `token_type_ids` es un tensor de ceros.
- [ ] El modelo está en modo `eval()` y el forward se ejecuta dentro de `torch.no_grad()`.
- [ ] El checkpoint cargado es `clasificador_texto_v4.pt`.
