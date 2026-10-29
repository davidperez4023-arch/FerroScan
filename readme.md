# FerroScan

**Institución Universitaria ITM** — Entrega 3: Visión Artificial

**Docente:** Cristian Mateo Hernández González

**Integrantes:**

* David Perez

* Carlos Andres Llano

## Descripción del Producto (Start-Up B2B)

**FerroScan** es un sistema de control de calidad industrial basado en **Visión Artificial** (sin redes neuronales convolucionales / Deep Learning) que detecta y clasifica en tiempo real los 6 defectos superficiales críticos del acero laminado en caliente sobre la base de datos **NEU Surface Defect Database** (1,800 imágenes).

## Pipeline Técnico Implementado

1. **Extracción de Características (`1_extraccion_ferroscan.py`):**

2. **Benchmarking de Clasificadores (`2_benchmark_ferroscan.py`):**

## Estructura del Repositorio

```
FerroScan/
├── 1_extraccion_ferroscan.py       # Script 1: Extracción HOG, LBP, GLCM y Fusión Z-Score
├── 2_benchmark_ferroscan.py        # Script 2: Comparativa k-NN vs SVM vs Random Forest
├── FerroScan.ipynb                 # Jupyter Notebook limpio con el pipeline unificado
├── resultados/                     # Gráficas generadas del Benchmark y Matriz de Confusión
└── README.md                       # Documentación del proyecto

```

## Cómo ejecutar el proyecto en Spyder (Python 3.10)

1. Descargar la base de datos [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) y descomprimir las imágenes dentro de la carpeta `dataset/`.

2. Ejecutar `1_extraccion_ferroscan.py` para extraer y normalizar los descriptores en `features/`.

3. Ejecutar `2_benchmark_ferroscan.py` para entrenar los modelos y generar las gráficas de desempeño.