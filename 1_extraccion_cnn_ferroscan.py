# -*- coding: utf-8 -*-
import os
import json
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ferroscan_common import (
    SEED, IMG_SIZE, TARGET_COL, cargar_imagen, crear_extractor, nombres_columnas
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings_cnn")

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMAGENES_ESPERADAS_POR_CLASE = 300  # NEU-DET: 6 clases x 300 = 1800 imágenes

CLASES_NEU = {
    0: {"nombre": "Crazing", "carpetas": ["crazing", "Cr"], "prefijo": "Cr"},
    1: {"nombre": "Inclusion", "carpetas": ["inclusion", "In"], "prefijo": "In"},
    2: {"nombre": "Patches", "carpetas": ["patches", "Pa"], "prefijo": "Pa"},
    3: {"nombre": "Pitted", "carpetas": ["pitted_surface", "PS", "pitted"], "prefijo": "PS"},
    4: {"nombre": "Rolled-in", "carpetas": ["rolled-in_scale", "RS", "rolled"], "prefijo": "RS"},
    5: {"nombre": "Scratches", "carpetas": ["scratches", "Sc"], "prefijo": "Sc"}
}
EXTENSIONES = ('.bmp', '.jpg', '.jpeg', '.png')

tf.keras.utils.set_random_seed(SEED)


# 1. Recolección de rutas y etiquetas (SIN tope silencioso por clase)
def obtener_rutas_y_etiquetas(dataset_dir):
    archivos = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(EXTENSIONES):
                archivos.append(os.path.join(root, f))

    rutas, etiquetas = [], []
    for file_path in sorted(archivos):
        filename = os.path.basename(file_path)
        parent = os.path.basename(os.path.dirname(file_path)).lower()

        clase_id = None
        for label_id, info in CLASES_NEU.items():
            if any(c.lower() == parent for c in info["carpetas"]) or \
               filename.lower().startswith(info["prefijo"].lower() + "_") or \
               filename.lower().startswith(info["carpetas"][0].lower()):
                clase_id = label_id
                break

        if clase_id is not None:
            rutas.append(file_path)
            etiquetas.append(CLASES_NEU[clase_id]["nombre"])

    return np.array(rutas), np.array(etiquetas)


def _md5(ruta):
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# 2. Verificación de integridad: conteo por clase, nombres repetidos y duplicados por contenido.
#    Las copias exactas se eliminan (se conserva la primera) para que no puedan caer
#    una en train y otra en test. Si un mismo contenido aparece con etiquetas distintas
#    es un error de etiquetado y se detiene la ejecución.
def verificar_dataset(rutas, etiquetas):
    problemas = []

    for info in CLASES_NEU.values():
        n = int(np.sum(etiquetas == info["nombre"]))
        if n != IMAGENES_ESPERADAS_POR_CLASE:
            problemas.append(f"Clase {info['nombre']}: {n} imágenes (se esperaban {IMAGENES_ESPERADAS_POR_CLASE})")

    nombres = [os.path.basename(r) for r in rutas]
    if len(set(nombres)) != len(nombres):
        problemas.append(f"Hay {len(nombres) - len(set(nombres))} nombres de archivo repetidos "
                         f"(posibles carpetas duplicadas)")

    grupos = {}
    for r, e in zip(rutas, etiquetas):
        grupos.setdefault(_md5(r), []).append((str(r), str(e)))

    conservar = set()
    eliminadas = 0
    for items in grupos.values():
        conservar.add(items[0][0])
        if len(items) > 1:
            if len({e for _, e in items}) > 1:
                problemas.append(f"Mismo contenido con etiquetas distintas: {[r for r, _ in items]}")
            else:
                eliminadas += len(items) - 1
                print(f"[AVISO] Imágenes idénticas ({items[0][1]}): "
                      f"{[os.path.basename(r) for r, _ in items]} -> se conserva solo "
                      f"{os.path.basename(items[0][0])}")

    if problemas:
        raise ValueError(
            "El dataset no pasó la verificación:\n - " + "\n - ".join(problemas) +
            "\nRevisa carpetas duplicadas (p. ej. copias aumentadas o train/validation repetidos)."
        )

    mascara = np.array([str(r) in conservar for r in rutas])
    print(f"Dataset verificado: {len(rutas)} imágenes encontradas, {eliminadas} duplicada(s) "
          f"eliminada(s) -> {int(mascara.sum())} imágenes únicas.")
    return rutas[mascara], etiquetas[mascara]


# 3. Pipeline tf.data.Dataset (mismo preprocesado que se usará en inferencia)
def crear_dataset_tf(file_paths):
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(cargar_imagen, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return ds  # sin shuffle: el orden de los embeddings coincide con las etiquetas


if __name__ == "__main__":
    list_file_paths, list_labels = obtener_rutas_y_etiquetas(DATASET_DIR)
    print(f"Total de imágenes detectadas: {len(list_file_paths)}")
    list_file_paths, list_labels = verificar_dataset(list_file_paths, list_labels)
    print(f"Imágenes únicas usadas: {len(list_file_paths)}")

    paths_train, paths_test, y_train, y_test = train_test_split(
        list_file_paths, list_labels, test_size=0.30, random_state=SEED, stratify=list_labels
    )

    train_ds = crear_dataset_tf(paths_train)
    test_ds = crear_dataset_tf(paths_test)

    extractor = crear_extractor()

    # 4. Extracción de embeddings (1,280 características por imagen)
    print("Extrayendo embeddings con EfficientNetB0 (Train y Test)...")
    features_train = extractor.predict(train_ds, verbose=1)
    features_test = extractor.predict(test_ds, verbose=1)

    assert len(features_train) == len(y_train) and len(features_test) == len(y_test)

    # 5. Estructuración tabular (f_0 ... f_1279 + etiqueta) y guardado
    cols = nombres_columnas(features_train.shape[1])

    df_train = pd.DataFrame(features_train, columns=cols)
    df_train[TARGET_COL] = y_train

    df_test = pd.DataFrame(features_test, columns=cols)
    df_test[TARGET_COL] = y_test

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_train.to_csv(os.path.join(OUTPUT_DIR, "embeddings_train.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "embeddings_test.csv"), index=False)
    pd.DataFrame({'ruta': paths_test, TARGET_COL: y_test}).to_csv(
        os.path.join(OUTPUT_DIR, "rutas_test.csv"), index=False
    )

    # 6. Metadatos para reproducibilidad
    meta = {
        "seed": SEED,
        "img_size": list(IMG_SIZE),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "tensorflow": tf.__version__,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataFrames guardados en: {OUTPUT_DIR}")
    print(f"Train tabular: {df_train.shape} | Test tabular: {df_test.shape}")