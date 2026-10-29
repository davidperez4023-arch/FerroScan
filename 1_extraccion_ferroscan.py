import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "features")

CLASES_NEU = {
    0: {"carpetas": ["crazing", "Cr"], "prefijo": "Cr"},
    1: {"carpetas": ["inclusion", "In"], "prefijo": "In"},
    2: {"carpetas": ["patches", "Pa"], "prefijo": "Pa"},
    3: {"carpetas": ["pitted_surface", "PS", "pitted"], "prefijo": "PS"},
    4: {"carpetas": ["rolled-in_scale", "RS", "rolled"], "prefijo": "RS"},
    5: {"carpetas": ["scratches", "Sc"], "prefijo": "Sc"}
}
NOMBRES_CORTOS = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled-in", "Scratches"]
EXTENSIONES = ('.bmp', '.jpg', '.jpeg', '.png')


# 1. Carga y normalización de las 6 clases (128x128)
def cargar_dataset_neu(dataset_dir, img_size=(128, 128), max_por_clase=300):
    todos_los_archivos = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(EXTENSIONES):
                todos_los_archivos.append(os.path.join(root, f))

    X_list, y_list = [], []
    conteo = {k: 0 for k in CLASES_NEU.keys()}

    for file_path in sorted(todos_los_archivos):
        filename = os.path.basename(file_path)
        parent = os.path.basename(os.path.dirname(file_path)).lower()

        clase_id = None
        for label_id, info in CLASES_NEU.items():
            if any(c.lower() == parent for c in info["carpetas"]) or \
               filename.lower().startswith(info["prefijo"].lower() + "_") or \
               filename.lower().startswith(info["carpetas"][0].lower()):
                clase_id = label_id
                break

        if clase_id is not None and conteo[clase_id] < max_por_clase:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size).astype("float32") / 255.0
                X_list.append(img)
                y_list.append(clase_id)
                conteo[clase_id] += 1

    return np.array(X_list), np.array(y_list, dtype=np.int32)


# 2. Extractores clásicos: HOG, LBP y GLCM
def extraer_hog(imagenes):
    hog_features = []
    for img in imagenes:
        img_gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        feat = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(feat)
    return np.array(hog_features)


def extraer_lbp(imagenes, P=8, R=1):
    lbp_features = []
    for img in imagenes:
        img_gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(img_gray, P=P, R=R, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float32") / (hist.sum() + 1e-7)
        lbp_features.append(hist)
    return np.array(lbp_features)


def extraer_glcm(imagenes, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm_features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for img in imagenes:
        img_gray_q = (cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY) // 4).astype(np.uint8)
        glcm = graycomatrix(img_gray_q, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)
        glcm_features.append([val for prop in props for val in graycoprops(glcm, prop).ravel()])
    return np.array(glcm_features)


if __name__ == "__main__":
    X, y = cargar_dataset_neu(DATASET_DIR)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    print(f"Dataset listo -> Train: {X_train.shape}, Validación: {X_val.shape}")

    print("Extrayendo HOG, LBP y GLCM...")
    X_train_hog, X_val_hog = extraer_hog(X_train), extraer_hog(X_val)
    X_train_lbp, X_val_lbp = extraer_lbp(X_train), extraer_lbp(X_val)
    X_train_glcm, X_val_glcm = extraer_glcm(X_train), extraer_glcm(X_val)

    # 3. Fusión y normalización Z-Score (StandardScaler)
    X_train_fusion = np.hstack((X_train_hog, X_train_lbp, X_train_glcm))
    X_val_fusion = np.hstack((X_val_hog, X_val_lbp, X_val_glcm))

    scaler_fusion = StandardScaler()
    X_train_fusion_scaled = scaler_fusion.fit_transform(X_train_fusion)
    X_val_fusion_scaled = scaler_fusion.transform(X_val_fusion)

    # 4. Gráfica 1: Mapas visuales HOG, LBP y GLCM en las 6 clases
    indices_clases = [np.where(y_val == c)[0][0] for c in range(6)]
    fig1, axes1 = plt.subplots(4, 6, figsize=(16, 10))

    for col, idx in enumerate(indices_clases):
        img_gray = cv2.cvtColor((X_val[idx] * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        _, hog_vis = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        lbp_vis = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
        glcm_vis = graycomatrix((img_gray // 4).astype(np.uint8), distances=[1], angles=[0],
                                levels=64, symmetric=True, normed=True)[:, :, 0, 0]

        axes1[0, col].imshow(X_val[idx])
        axes1[0, col].set_title(NOMBRES_CORTOS[col], fontweight='bold')
        axes1[1, col].imshow(hog_vis, cmap='inferno')
        axes1[2, col].imshow(lbp_vis, cmap='gray')
        axes1[3, col].imshow(glcm_vis, cmap='viridis')
        for r in range(4):
            axes1[r, col].axis('off')

    plt.figure(fig1.number)
    plt.tight_layout()
    plt.show()

    # 5. Gráfica 2: Firma Fusionada Z-Score (Scratches vs Pitted Surface)
    idx_sc, idx_ps = indices_clases[5], indices_clases[3]
    fusion_sc_vis = X_val_fusion_scaled[idx_sc].reshape(1, -1)
    fusion_ps_vis = X_val_fusion_scaled[idx_ps].reshape(1, -1)

    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 6))

    axes2[0, 0].imshow(X_val[idx_sc])
    axes2[0, 0].set_title("Scratches (Entrada)")
    axes2[0, 0].axis('off')

    axes2[0, 1].imshow(X_val[idx_ps])
    axes2[0, 1].set_title("Pitted Surface (Entrada)")
    axes2[0, 1].axis('off')

    axes2[1, 0].imshow(fusion_sc_vis, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2)
    axes2[1, 0].set_title(f"Firma Fusionada ({fusion_sc_vis.shape[1]} dims) - Scratches")
    axes2[1, 0].set_yticks([])
    axes2[1, 0].set_xlabel("Índices de características (Normalizadas Z-Score)")

    axes2[1, 1].imshow(fusion_ps_vis, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2)
    axes2[1, 1].set_title(f"Firma Fusionada ({fusion_ps_vis.shape[1]} dims) - Pitted Surface")
    axes2[1, 1].set_yticks([])
    axes2[1, 1].set_xlabel("Índices de características (Normalizadas Z-Score)")

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.show()

    # 6. Guardado de características escaladas en .npz
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for nombre, X_tr, X_va in [
        ("features_hog.npz", X_train_hog, X_val_hog),
        ("features_lbp.npz", X_train_lbp, X_val_lbp),
        ("features_glcm.npz", X_train_glcm, X_val_glcm)
    ]:
        sc = StandardScaler()
        np.savez_compressed(os.path.join(OUTPUT_DIR, nombre),
                            X_train=sc.fit_transform(X_tr), X_val=sc.transform(X_va),
                            y_train=y_train, y_val=y_val)

    np.savez_compressed(os.path.join(OUTPUT_DIR, "features_fusion.npz"),
                        X_train=X_train_fusion_scaled, X_val=X_val_fusion_scaled,
                        y_train=y_train, y_val=y_val)

    print("Características guardadas exitosamente en:", OUTPUT_DIR)