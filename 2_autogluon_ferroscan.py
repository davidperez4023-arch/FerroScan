# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from ferroscan_common import (
    SEED, TARGET_COL, crear_extractor, nombres_columnas, embedding_de_imagen
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings_cnn")
MODEL_DIR = os.path.join(BASE_DIR, "autogluon_ferroscan_model")
FIG_DIR = os.path.join(BASE_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


def intervalo_wilson(aciertos, n, z=1.96):
    """IC 95% de Wilson para una proporción (accuracy)."""
    p = aciertos / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    margen = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centro - margen, centro + margen


# 1. Carga de los DataFrames tabulares de embeddings
df_train = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "embeddings_train.csv"))
df_test = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "embeddings_test.csv"))
df_rutas_test = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "rutas_test.csv"))

# 2. Entrenamiento de la cabeza de decisión con AutoGluon
#    Se optimiza accuracy, que es la métrica que se reporta.
#    (El ROC-AUC se calcula aparte como métrica complementaria.)
print(f"--- AutoGluon entrenando cabeza de decisión multiclase para: {TARGET_COL} ---")
predictor = TabularPredictor(
    label=TARGET_COL,
    problem_type='multiclass',
    eval_metric='accuracy',
    path=MODEL_DIR
).fit(
    train_data=df_train,
    time_limit=300
)

# Clases en el orden interno del predictor (en vez de una lista escrita a mano)
CLASES = list(predictor.class_labels)

# 3. Leaderboard y evaluación en el conjunto de prueba
leaderboard = predictor.leaderboard(df_test, silent=True)
print("\n" + "=" * 95)
print("LEADERBOARD AUTOGLUON - FERROSCAN (CONJUNTO DE PRUEBA, métrica: accuracy)")
print("=" * 95)
print(leaderboard[['model', 'score_test', 'score_val', 'pred_time_test', 'fit_time']].to_string(index=False))

X_test = df_test.drop(columns=[TARGET_COL])
y_true = df_test[TARGET_COL].values
y_pred = predictor.predict(X_test).values
y_proba = predictor.predict_proba(X_test)[CLASES].values
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # evita fallos por redondeo en float32

acc = accuracy_score(y_true, y_pred)
aciertos = int((y_true == y_pred).sum())
ic_low, ic_high = intervalo_wilson(aciertos, len(y_true))
roc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=CLASES)

print(f"\nAccuracy Global en Prueba: {acc * 100:.2f}% ({aciertos}/{len(y_true)}) "
      f"| IC 95% (Wilson): [{ic_low * 100:.2f}%, {ic_high * 100:.2f}%]")
print(f"ROC-AUC Macro (OvR): {roc_macro:.4f}\n")
print("Reporte de Clasificación por Defecto:")
print(classification_report(y_true, y_pred, labels=CLASES, target_names=CLASES, digits=4))

# 4. Gráfica 1: Matriz de Confusión (Conjunto de Prueba)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred, labels=CLASES)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES).plot(
    cmap='Blues', ax=ax_cm, values_format='d'
)
plt.title(f"Matriz de Confusión (conjunto de prueba, n={len(y_true)}) - "
          f"EfficientNetB0 + AutoGluon\nAccuracy: {acc * 100:.2f}%")
plt.xlabel("Defecto Predicho")
plt.ylabel("Defecto Real")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "matriz_confusion.png"), dpi=200)
plt.show()

# 5. Gráfica 2: Curvas ROC-AUC Multiclase (One-vs-Rest)
y_true_bin = label_binarize(y_true, classes=CLASES)
plt.figure(figsize=(9, 7))
for i, clase in enumerate(CLASES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{clase} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Azar (AUC = 0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title(f"Curvas ROC-AUC Multiclase - FerroScan (AUC Macro = {roc_macro:.4f})")
plt.legend(loc="lower right")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "curvas_roc.png"), dpi=200)
plt.show()

# 6. Comprobación en tiempo real, con la MISMA ruta de preprocesado del entrenamiento
extractor_rt = crear_extractor()

# Mantener los modelos en memoria evita recargarlos en cada predicción
if hasattr(predictor, "persist"):
    predictor.persist()


def predecir_imagen(ruta_imagen, extractor_cnn, modelo_autogluon):
    """Devuelve (clase, confianza %, ms_cnn, ms_cabeza)."""
    t0 = time.perf_counter()
    embedding = embedding_de_imagen(ruta_imagen, extractor_cnn)
    t1 = time.perf_counter()
    df_sample = pd.DataFrame(embedding, columns=nombres_columnas(embedding.shape[1]))
    probs = modelo_autogluon.predict_proba(df_sample).iloc[0]
    t2 = time.perf_counter()
    return probs.idxmax(), probs.max() * 100.0, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def comprobar_en_tiempo_real(ruta_imagen, extractor_cnn, modelo_autogluon, etiqueta_real=None):
    pred_clase, confianza, ms_cnn, ms_cabeza = predecir_imagen(ruta_imagen, extractor_cnn, modelo_autogluon)

    # cv2 solo se usa para mostrar la imagen, no para el preprocesado del modelo
    img_rgb = cv2.cvtColor(cv2.imread(ruta_imagen), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    titulo = f"Predicción: {pred_clase} ({confianza:.1f}%)\nCNN {ms_cnn:.1f} ms + cabeza {ms_cabeza:.1f} ms"
    if etiqueta_real:
        titulo = f"Real: {etiqueta_real}\n" + titulo
    plt.title(titulo, fontsize=11, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"[TIEMPO REAL] {os.path.basename(ruta_imagen)} -> {pred_clase} ({confianza:.2f}%) "
          f"| CNN {ms_cnn:.1f} ms + cabeza {ms_cabeza:.1f} ms")


def medir_latencia(rutas, extractor_cnn, modelo_autogluon, n=30):
    """Latencia por imagen tras un calentamiento (la primera llamada no se cuenta)."""
    predecir_imagen(rutas[0], extractor_cnn, modelo_autogluon)  # warm-up
    cnn, cab = [], []
    for r in rutas[:n]:
        _, _, a, b = predecir_imagen(r, extractor_cnn, modelo_autogluon)
        cnn.append(a)
        cab.append(b)
    total = np.array(cnn) + np.array(cab)
    print(f"\nLatencia sobre {len(total)} imágenes (tras calentamiento):")
    print(f"  CNN:    media {np.mean(cnn):.1f} ms")
    print(f"  Cabeza: media {np.mean(cab):.1f} ms")
    print(f"  Total:  media {total.mean():.1f} ms | mediana {np.median(total):.1f} ms "
          f"| p95 {np.percentile(total, 95):.1f} ms")


# Imágenes nuevas por línea de comandos: python 2_autogluon_ferroscan.py img1.bmp img2.bmp
if len(sys.argv) > 1:
    for ruta in sys.argv[1:]:
        comprobar_en_tiempo_real(ruta, extractor_rt, predictor)
else:
    muestras_demo = df_rutas_test.sample(n=3, random_state=SEED)
    for _, fila in muestras_demo.iterrows():
        comprobar_en_tiempo_real(fila['ruta'], extractor_rt, predictor, etiqueta_real=fila[TARGET_COL])

medir_latencia(df_rutas_test['ruta'].tolist(), extractor_rt, predictor)