import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
FEATURES_DIR = os.path.join(BASE_DIR, "features")

feature_files = {
    'HOG': 'features_hog.npz',
    'LBP': 'features_lbp.npz',
    'GLCM': 'features_glcm.npz',
    'FUSIÓN (HOG+LBP+GLCM)': 'features_fusion.npz'
}

NOMBRES_CLASES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled-in", "Scratches"]

# 1. Definición de clasificadores clásicos
models = {
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1),
    'SVM (RBF)': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

# 2. Entrenamiento, evaluación multiclase (macro) y tiempos de negocio
results = []
best_f1 = -1.0
best_combo = {}

for feat_name, file_name in feature_files.items():
    data = np.load(os.path.join(FEATURES_DIR, file_name))
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    for model_name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0

        t0_pred = time.time()
        y_pred = model.predict(X_val)
        ms_por_img = ((time.time() - t0_pred) / len(y_val)) * 1000.0

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        results.append({
            'Descriptor': feat_name,
            'Modelo': model_name,
            'Dimensiones': X_train.shape[1],
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Train (s)': t_train,
            'Inferencia (ms/img)': ms_por_img
        })

        if f1 > best_f1:
            best_f1 = f1
            best_combo = {'Descriptor': feat_name, 'Modelo': model_name,
                          'y_val': y_val, 'y_pred': y_pred, 'Accuracy': acc}

df_results = pd.DataFrame(results)
print("=" * 95)
print("TABLA CONSOLIDADA DE BENCHMARK - FERROSCAN (NEU-DET)")
print("=" * 95)
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nMejor sistema: {best_combo['Descriptor']} + {best_combo['Modelo']}")
print(classification_report(best_combo['y_val'], best_combo['y_pred'], target_names=NOMBRES_CLASES, digits=4))

# 3. Gráfica comparativa de F1-Score y Matriz de Confusión del mejor modelo
df_pivot = df_results.pivot(index='Descriptor', columns='Modelo', values='F1-Score').loc[list(feature_files.keys())]
ax = df_pivot.plot(kind='bar', figsize=(11, 6), colormap='Set2', edgecolor='black')
plt.title("Comparativa de Desempeño (F1-Score) según Extractor y Algoritmo ML", fontsize=13)
plt.ylabel("F1-Score")
plt.ylim(0, 1.08)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(best_combo['y_val'], best_combo['y_pred'])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NOMBRES_CLASES).plot(cmap='Blues', ax=ax_cm, values_format='d')
plt.title(f"Matriz de Confusión - {best_combo['Descriptor']} + {best_combo['Modelo']}")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()