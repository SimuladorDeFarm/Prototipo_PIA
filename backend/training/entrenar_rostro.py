"""
Entrenamiento del clasificador de emociones por rostro (Random Forest).

Entrena sobre los features extraídos por Py-Feat (CSV pyfeat_features_v5.csv),
siguiendo la metodología descrita en Docs/pipeline_rostro.md.

Decisiones de este entrenamiento:
- Taxonomía de 7 clases: se descarta 'contempt' (no pertenece a la taxonomía
  unificada del proyecto, la misma que usa el módulo de voz).
- Conjunto de entrenamiento = particiones Train + Train_balanced. Ninguna de las
  dos por separado cubre las 7 clases (Train no tiene 'neutral',
  Train_balanced no tiene 'surprise'); combinadas sí.
- Evaluación sobre la partición Test.
- Se aplican los filtros de calidad (QC) del pipeline antes de entrenar.

Uso:
    python entrenar_rostro.py --csv ../../../pyfeat_features_v5.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Definición de features (orden EXACTO de pipeline_rostro.md) ---
AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43",
]
LANDMARK_COLS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
FEATURE_COLS = AU_COLS + ["FaceScore"]   # 21 features

# --- Umbrales QC (Docs/pipeline_rostro.md) ---
FACESCORE_MIN = 0.90
POSE_MAX_RAD = np.deg2rad(45.0)

# Taxonomía unificada de 7 clases (se descarta 'contempt')
CLASES_VALIDAS = ["neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"]
PARTICIONES_TRAIN = ["Train", "Train_balanced"]
PARTICION_TEST = "Test"


def cargar_y_etiquetar(ruta_csv: Path) -> pd.DataFrame:
    """Carga el CSV y deriva 'particion' y 'emocion' desde la columna 'imagen'."""
    df = pd.read_csv(ruta_csv)
    partes = df["imagen"].str.split("/", expand=True)
    df["particion"] = partes[0]
    df["emocion"] = partes[1].str.lower()  # normaliza Anger/anger -> anger
    return df


def filtros_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los filtros de calidad del pipeline de inferencia."""
    n0 = len(df)

    # 1. Rostro detectado con confianza suficiente
    df = df[df["face_detected"] == True]                      # noqa: E712
    df = df[df["FaceScore"].notna() & (df["FaceScore"] >= FACESCORE_MIN)]
    n1 = len(df)

    # 2. Pose dentro de ±45° (en radianes)
    pose_ok = (
        df["Yaw"].abs() <= POSE_MAX_RAD
    ) & (df["Pitch"].abs() <= POSE_MAX_RAD) & (df["Roll"].abs() <= POSE_MAX_RAD)
    df = df[pose_ok]
    n2 = len(df)

    # 3. Landmarks sin NaN
    df = df[~df[LANDMARK_COLS].isna().any(axis=1)]
    n3 = len(df)

    # 4. AUs no degenerados (no todos 0 ni todos 1)
    aus = df[AU_COLS]
    degenerados = (aus == 0).all(axis=1) | (aus == 1).all(axis=1)
    df = df[~degenerados]
    n4 = len(df)

    print(f"  QC: {n0} -> FaceScore {n1} -> pose {n2} -> landmarks {n3} -> AUs {n4}")
    return df


def main():
    parser = argparse.ArgumentParser()
    aqui = Path(__file__).resolve().parent
    parser.add_argument(
        "--csv",
        type=Path,
        default=aqui.parent.parent.parent / "pyfeat_features_v5.csv",
        help="Ruta al CSV de features de Py-Feat.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=aqui.parent / "models" / "clasificador_rostro.joblib",
    )
    args = parser.parse_args()

    print(f"Cargando {args.csv} ...")
    df = cargar_y_etiquetar(args.csv)
    print(f"Filas totales: {len(df)}")

    # Descartar contempt (fuera de la taxonomía de 7 clases)
    df = df[df["emocion"].isin(CLASES_VALIDAS)]
    print(f"Tras descartar 'contempt' (7 clases): {len(df)}")

    # Split por partición
    df_train = df[df["particion"].isin(PARTICIONES_TRAIN)].copy()
    df_test = df[df["particion"] == PARTICION_TEST].copy()
    print(f"\nEntrenamiento (Train + Train_balanced): {len(df_train)}")
    df_train = filtros_qc(df_train)
    print(f"Test: {len(df_test)}")
    df_test = filtros_qc(df_test)

    print("\nDistribución de clases en entrenamiento:")
    print(df_train["emocion"].value_counts().to_string())

    X_train = df_train[FEATURE_COLS].values.astype(float)
    y_train = df_train["emocion"].values
    X_test = df_test[FEATURE_COLS].values.astype(float)
    y_test = df_test["emocion"].values

    # Hiperparámetros documentados (Docs/pipeline_rostro.md)
    print("\nEntrenando Random Forest...")
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        criterion="gini",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    print("\n=== Evaluación sobre Test ===")
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión (filas=real, cols=pred):")
    print("Clases:", list(modelo.classes_))
    print(confusion_matrix(y_test, y_pred, labels=modelo.classes_))

    # --- Guardar BUNDLE autodescriptivo ---
    # El backend lee el contrato (features, clases, umbrales) desde aquí, no del
    # código. Para cambiar el modelo basta con guardar otro bundle con este mismo
    # formato; el programa de inferencia sigue funcionando sin modificaciones.
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "modelo": modelo,              # cualquier clasificador sklearn (predict/predict_proba/classes_)
        "features": FEATURE_COLS,      # columnas de entrada y su ORDEN exacto
        "clases": list(modelo.classes_),
        "umbral_facescore": FACESCORE_MIN,
        "pose_max_grados": 45.0,
        "version": "rostro-v5",
    }
    joblib.dump(bundle, args.salida)

    meta = {
        "tipo": type(modelo).__name__,
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "clases": list(modelo.classes_),
        "umbral_facescore": FACESCORE_MIN,
        "pose_max_grados": 45.0,
        "formato_artefacto": "bundle joblib: {modelo, features, clases, umbral_facescore, pose_max_grados, version}",
        "hiperparametros": {
            "n_estimators": 300,
            "max_features": "sqrt",
            "criterion": "gini",
            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "entrenamiento": {
            "particiones": PARTICIONES_TRAIN,
            "n_train": int(len(df_train)),
            "n_test": int(len(df_test)),
            "csv": args.csv.name,
        },
    }
    meta_path = args.salida.with_name("clasificador_rostro_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModelo guardado en: {args.salida}")
    print(f"Metadatos en:       {meta_path}")


if __name__ == "__main__":
    main()
