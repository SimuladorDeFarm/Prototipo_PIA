"""Extracción de embeddings con HuBERT congelado.

Para cada clip del dataset final se hace un forward pass por HuBERT (sin
segmentar: el audio completo), se aplica mean pooling sobre la dimensión
temporal y se obtiene un único vector de 768 dimensiones que representa el clip.

Los embeddings se guardan en disco como caché permanente (`embeddings.npz`):
se calculan una sola vez y los experimentos de la cabeza clasificadora cargan
desde ahí, sin volver a pasar el audio por HuBERT en cada ejecución.

El audio de entrada ya viene listo (mono, 16 kHz, normalizado, sin silencio de
inicio/fin) y las etiquetas de 7 clases vienen del índice `etiquetas.csv`.
"""

import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import config


def _leer_indice(csv_path):
    """Lee etiquetas.csv → lista de dicts (actor, archivo, emocion)."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def cargar_audio(ruta):
    """Carga un wav como float32 mono y verifica el sample rate."""
    data, sr = sf.read(ruta, dtype="float32")
    if data.ndim > 1:                       # defensivo: el dataset ya es mono
        data = data.mean(axis=1).astype(np.float32)
    if sr != config.SAMPLE_RATE_TARGET:
        raise ValueError(
            f"{ruta}: sample rate {sr}, se esperaba {config.SAMPLE_RATE_TARGET}")
    return data


@torch.no_grad()
def extraer_embedding(modelo, dispositivo, waveform):
    """Forward pass + mean pooling temporal → vector [768] (numpy float32)."""
    entrada = torch.from_numpy(waveform).unsqueeze(0).to(dispositivo)  # [1, T]
    salida = modelo(entrada).last_hidden_state                         # [1, F, 768]
    embedding = salida.mean(dim=1).squeeze(0)                          # [768]
    return embedding.cpu().numpy().astype(np.float32)


def extraer_embeddings(modelo, dispositivo, dataset_dir=None,
                       csv_path=None, salida=None, forzar=False):
    """Extrae y guarda los embeddings de todos los clips del dataset final.

    Devuelve la ruta del archivo .npz generado (o existente si ya estaba en
    caché y `forzar` es False).
    """
    dataset_dir = Path(dataset_dir or config.DATASET_FINAL_PATH)
    csv_path = Path(csv_path or config.ETIQUETAS_FINAL_CSV)
    salida = Path(salida or config.EMBEDDINGS_FILE)

    if salida.exists() and not forzar:
        print(f"✅ Embeddings ya en caché: {salida} (usar --forzar para recalcular)")
        return salida
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el índice de etiquetas: {csv_path}")

    filas = _leer_indice(csv_path)
    total = len(filas)
    print("=== EXTRACCIÓN DE EMBEDDINGS (HuBERT congelado) ===")
    print(f"Clips a procesar : {total}")
    print(f"Dispositivo      : {dispositivo}")
    print(f"Salida           : {salida}\n")

    X = np.empty((total, config.EMBEDDING_DIM), dtype=np.float32)
    actores, archivos, emociones = [], [], []

    for i, fila in enumerate(filas):
        ruta = dataset_dir / fila["actor"] / fila["archivo"]
        waveform = cargar_audio(ruta)
        X[i] = extraer_embedding(modelo, dispositivo, waveform)

        actores.append(fila["actor"])
        archivos.append(fila["archivo"])
        emociones.append(fila["emocion"])

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  procesados {i + 1}/{total}")

    os.makedirs(salida.parent, exist_ok=True)
    np.savez(
        salida,
        X=X,
        actor=np.array(actores),
        archivo=np.array(archivos),
        emocion=np.array(emociones),
    )

    _imprimir_resumen(X, emociones, salida)
    return salida


def _imprimir_resumen(X, emociones, salida):
    print("\n--- Resumen ---")
    print(f"Matriz de embeddings : {X.shape}  (clips × dimensión)")
    print(f"Tipo                 : {X.dtype}")
    print(f"Guardado en          : {salida}")

    print("\nDistribución por emoción:")
    valores, conteos = np.unique(np.array(emociones), return_counts=True)
    orden = {e: i for i, e in enumerate(config.EMOCIONES_7)}
    for emocion in sorted(valores, key=lambda e: orden.get(e, 99)):
        print(f"  {emocion:<10}: {conteos[list(valores).index(emocion)]}")

    assert X.shape[1] == config.EMBEDDING_DIM, "⚠️ Dimensión de embedding inesperada"
    assert not np.isnan(X).any(), "⚠️ Hay NaN en los embeddings"
    print("\n✅ Embeddings extraídos y guardados en caché correctamente")
