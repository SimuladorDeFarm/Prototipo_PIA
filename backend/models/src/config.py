"""Configuración del pipeline de preparación de datos del módulo de voz.

Todas las constantes y rutas viven aquí (nada hardcodeado en medio del código).
Cubre el trim de silencio y la fusión de etiquetas a 7 emociones.

La raíz de datos se resuelve desde la variable de entorno `VOZ_DATA_DIR`; si no
está definida, se usa la raíz del proyecto (donde viven las carpetas del dataset).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VOZ_DATA_DIR", PROJECT_ROOT))

# --- Rutas -----------------------------------------------------------------
# Entrada: dataset actual ("RMS ajustado"), un subdirectorio por actor.
DATASET_RMS_PATH = DATA_DIR / "dataset_rms"
# Salida: dataset con silencio de inicio/fin recortado (misma estructura).
DATASET_TRIM_PATH = DATA_DIR / "dataset_trim"
# Reporte de cuánto se recortó por archivo (para detectar clips atípicos).
TRIM_REPORT_CSV = DATA_DIR / "trim_report.csv"

# --- Audio -----------------------------------------------------------------
SAMPLE_RATE_TARGET = 16000   # mono + 16 kHz, heredado del dataset RMS

# --- Parámetros de detección de silencio -----------------------------------
# Umbral en dB por debajo del pico: los frames más silenciosos que esto se
# consideran silencio. Solo se recorta el silencio de inicio y fin (no interno).
TRIM_TOP_DB = 30.0
FRAME_LENGTH = 2048   # tamaño de ventana para medir energía por frame
HOP_LENGTH = 512      # salto entre frames

# Umbral para marcar como sospechoso un clip que quede muy corto tras el trim.
MIN_DURACION_S = 0.5

# --- Fusión de etiquetas (taxonomía de 7 emociones) ------------------------
# Entrada: dataset con silencio recortado. Salida: dataset final con las
# etiquetas ya fusionadas (calmado → neutral) en un índice CSV.
DATASET_FINAL_PATH = DATA_DIR / "dataset_final"
ETIQUETAS_FINAL_CSV = DATASET_FINAL_PATH / "etiquetas.csv"

# El nombre de archivo RAVDESS codifica la emoción en su tercer campo:
#   03-01-EM-IN-ST-RP-AC.wav   (EM = código de emoción)
# Mapa código RAVDESS → emoción de la taxonomía unificada de 7 clases.
# "calmado" (02) se fusiona con "neutral" (01): ambos quedan como "neutral".
MAPA_EMOCION_RAVDESS = {
    "01": "neutral",   # neutral
    "02": "neutral",   # calmado → fusionado con neutral
    "03": "joy",       # alegría
    "04": "sadness",   # tristeza
    "05": "anger",     # enojo
    "06": "fear",      # miedo
    "07": "disgust",   # disgusto
    "08": "surprise",  # sorpresa
}

# Las 7 emociones finales, en el orden de la taxonomía unificada (sección 7).
EMOCIONES_7 = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

# --- Modelo: HuBERT pre-entrenado ------------------------------------------
# HuBERT base pre-entrenado en inglés (LibriSpeech 960h). Se usa congelado:
# solo como extractor de embeddings, sin entrenar sus parámetros.
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
EMBEDDING_DIM = 768   # dimensión del embedding de HuBERT base

# --- Extracción de embeddings ----------------------------------------------
# Caché permanente de embeddings (un vector de 768 por clip). Se calcula una
# sola vez; los experimentos posteriores cargan desde aquí sin recalcular.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npz"

# --- Reproducibilidad ------------------------------------------------------
# Semilla fija para que el split y el entrenamiento sean reproducibles.
SEED = 42

# --- Split por actor (en memoria) ------------------------------------------
# Proporción de actores que va a validación y prueba. El split se hace por
# ACTOR (un actor entero cae en un solo split) para no mezclar voces entre
# splits, y estratificado por género para mantener el balance 12 H / 12 M.
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- Cabeza clasificadora --------------------------------------------------
# Una sola cabeza: capa densa oculta + dropout → logits de las 7 clases.
HIDDEN_DIM = 256
DROPOUT = 0.3
NUM_CLASSES = len(EMOCIONES_7)   # 7

# --- Entrenamiento ---------------------------------------------------------
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# --- Checkpoint ------------------------------------------------------------
# Se guarda el mejor modelo según F1 macro en validación.
MODELOS_DIR = DATA_DIR / "modelos"
CHECKPOINT_FILE = MODELOS_DIR / "clasificador_voz.pt"
