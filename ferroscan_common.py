# -*- coding: utf-8 -*-
"""
Utilidades compartidas por los scripts de FerroScan.

Un solo preprocesado para entrenamiento e inferencia evita el desajuste
(train/serve skew) que había entre tf.image.resize y cv2.resize.
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (256, 256)
SEED = 42
NOMBRES_CLASES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled-in", "Scratches"]
TARGET_COL = "clase"


def cargar_imagen(ruta):
    """Decodifica, redimensiona y preprocesa una imagen (acepta str o tf.string)."""
    img = tf.io.read_file(ruta)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    # En EfficientNet de Keras la normalización va dentro del modelo y
    # preprocess_input es un paso neutro; se deja aquí por compatibilidad
    # entre versiones, y en UN solo lugar.
    return preprocess_input(tf.cast(img, tf.float32))


def crear_extractor():
    """EfficientNetB0 congelada (ImageNet) que devuelve 1280 características."""
    modelo = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    modelo.trainable = False
    return modelo


def nombres_columnas(n_features):
    return [f"f_{i}" for i in range(n_features)]


def embedding_de_imagen(ruta, extractor):
    """Embedding (1, 1280) de una sola imagen, con la misma ruta de preprocesado del entrenamiento."""
    x = tf.expand_dims(cargar_imagen(ruta), axis=0)
    return extractor(x, training=False).numpy()
