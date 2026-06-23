"""Carga del modelo HuBERT pre-entrenado (congelado).

HuBERT se usa únicamente como extractor de embeddings: se carga el modelo base
pre-entrenado y se congelan TODOS sus parámetros (`requires_grad = False`), de
modo que durante el entrenamiento de la cabeza clasificadora sus pesos no se
modifican. El modelo se pone en modo evaluación (`eval()`) porque no se entrena.

La primera ejecución descarga los pesos desde Hugging Face y los deja en caché
local; las siguientes ejecuciones cargan desde caché sin volver a descargar.
"""

import torch
from transformers import HubertModel

from . import config


def seleccionar_dispositivo(device=None):
    """Devuelve el dispositivo a usar: el indicado, o GPU si hay, si no CPU."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def congelar_parametros(modelo):
    """Congela todos los parámetros del modelo (no se entrenan)."""
    for parametro in modelo.parameters():
        parametro.requires_grad = False
    modelo.eval()
    return modelo


def cargar_hubert(nombre=None, device=None, verbose=True):
    """Carga HuBERT pre-entrenado, lo congela y lo deja listo para inferencia.

    Devuelve (modelo, dispositivo). Con `verbose=True` imprime el progreso y la
    verificación de que el modelo quedó congelado; con `verbose=False` carga en
    silencio (útil en inferencia).
    """
    nombre = nombre or config.HUBERT_MODEL_NAME
    dispositivo = seleccionar_dispositivo(device)

    if verbose:
        print("=== CARGA DEL MODELO HuBERT ===")
        print(f"Modelo      : {nombre}")
        print(f"Dispositivo : {dispositivo}")
        print("Descargando / cargando pesos pre-entrenados... "
              "(la primera vez puede tardar)")

    modelo = HubertModel.from_pretrained(nombre)
    modelo = congelar_parametros(modelo)
    modelo.to(dispositivo)

    if verbose:
        print("✅ Pesos cargados")
        _verificar_modelo(modelo)
    return modelo, dispositivo


def _verificar_modelo(modelo):
    """Comprueba que el modelo está congelado y reporta su tamaño."""
    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    dim_oculta = modelo.config.hidden_size

    print("\n--- Verificación ---")
    print(f"Parámetros totales    : {total:,}")
    print(f"Parámetros entrenables: {entrenables:,}")
    print(f"Modo                  : {'eval' if not modelo.training else 'train'}")
    print(f"Dimensión de embedding: {dim_oculta}")

    assert entrenables == 0, \
        f"⚠️ El modelo NO está completamente congelado ({entrenables} entrenables)"
    assert not modelo.training, "⚠️ El modelo no está en modo eval"
    assert dim_oculta == config.EMBEDDING_DIM, \
        f"⚠️ Dimensión inesperada: {dim_oculta} (esperado {config.EMBEDDING_DIM})"

    print("\n✅ HuBERT cargado, congelado (0 parámetros entrenables) y en modo eval")
