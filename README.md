# Proyecto MP6 — Predicción de ventas y segmentación de clientes en e-commerce

[![Status](https://img.shields.io/badge/status-en%20curso-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Tools](https://img.shields.io/badge/tools-Colab%20%7C%20scikit--learn%20%7C%20pandas-orange)]()

**Grupo H1** — Módulo MP6 (Proyecto de IA y Big Data) — Máster en IA y Big Data

---

## 📌 Descripción

Este repositorio contiene los entregables del proyecto del módulo MP6 del Máster en IA y Big Data. El proyecto se desarrolla sobre el dataset público **E-Commerce Data** (Kaggle / UCI), que recoge las transacciones de una tienda de regalos y artículos decorativos del Reino Unido entre diciembre de 2010 y diciembre de 2011.

El proyecto se divide en dos entregas complementarias:

- 🔮 **Entrega 1 — Regresión:** predicción de ventas diarias mediante modelos de series temporales y *machine learning*.
- 👥 **Entrega 2 — Clustering:** segmentación de clientes para acciones de *marketing* personalizadas.

---

## 📁 Estructura del repositorio

```
.
├── 01-entrega-regresion/
│   ├── datos/                   # Dataset original (comprimido)
│   ├── notebooks/               # Scripts de limpieza y modelos
│   └── memoria/                 # Memoria final (PDF)
│
├── 02-entrega-clustering/
│   ├── datos/                   # Dataset de clientes preprocesado
│   ├── notebooks/               # Scripts de los 4 modelos
│   └── memoria/                 # Memoria final (PDF + LaTeX)
│
└── docs/
    └── enunciados/              # Enunciados del proyecto
```

---

## 🔮 Entrega 1 — Regresión

**Objetivo:** predecir las ventas diarias de la tienda para apoyar la planificación de inventario y campañas comerciales.

### Modelos implementados

| Modelo | Tipo |
|---|---|
| Prophet | Series temporales descompuestas |
| ARIMA | Series temporales clásicas |
| Random Forest | *Ensemble* de árboles (*bagging*) |
| XGBoost | *Gradient Boosting* |
| LSTM | Red neuronal recurrente |
| Regresión Polinómica | Modelo lineal con términos polinómicos |

📄 **[Ver detalles de la entrega](./01-entrega-regresion/README.md)** • **[Memoria PDF](./01-entrega-regresion/memoria/)**

---

## 👥 Entrega 2 — Clustering

**Objetivo:** segmentar la base de clientes para personalizar estrategias de *marketing*, promociones y acciones comerciales.

### Modelos implementados

| Modelo | *Clusters* | Silueta |
|---|---|---|
| **K-Means** ⭐ | 2 | 0,5257 |
| Clustering Jerárquico | 2 | 0,7403 |
| DBSCAN | 4 | 0,3386 |
| HDBSCAN | 8 | 0,248 |

⭐ **Modelo seleccionado:** K-Means con preprocesado mejorado (PCA + LOF), por su balance entre simplicidad, interpretabilidad y métricas robustas. El Clustering Jerárquico queda como segunda opción válida, validando la estructura de K=2 desde un enfoque algorítmico distinto.

📄 **[Ver detalles de la entrega](./02-entrega-clustering/README.md)** • **[Memoria PDF](./02-entrega-clustering/memoria/)**

---

## 🛠️ Tecnologías utilizadas

- **Python 3.10+**
- **Google Colab** como entorno de ejecución
- **Pandas / NumPy** para manipulación de datos
- **Scikit-learn** para preprocesado, K-Means, DBSCAN, Random Forest
- **HDBSCAN** (librería externa) para *clustering* basado en densidad
- **SciPy** para *clustering* jerárquico y dendrograma
- **Prophet** (Meta) para series temporales
- **Statsmodels** para ARIMA
- **XGBoost** para *gradient boosting*
- **TensorFlow / Keras** para LSTM
- **Matplotlib / Seaborn** para visualización
- **LaTeX** para las memorias

---

## 🚀 Cómo ejecutar los scripts

Los scripts están diseñados para ejecutarse en **Google Colab** o en local con Python 3.10+. Para ejecutarlos:

1. Descomprime el dataset original (`01-entrega-regresion/datos/data.csv.zip`).
2. Abre cualquier script `.py` de la carpeta `notebooks/`.
3. Ajusta las rutas de los archivos de entrada/salida según tu entorno.
4. Ejecútalo en Colab o con Python en local.

Los scripts generan en local:
- Figuras (`.png`) con visualizaciones de cada modelo.
- CSVs con resultados, predicciones y métricas.

---

## 📖 Documentación

- **Memorias** (PDF y LaTeX) en cada carpeta `memoria/`.
- **Enunciados** del proyecto en `docs/enunciados/`.

---

## 👥 Equipo

**Grupo H1** — Módulo MP6 — Máster en IA y Big Data

- Paula Ruiz de la Parra (portavoz)
- Elena Roxana Dobroiu
- Julián David Dionisio Rey
- Javier Santos Rodriguez

## 📜 Licencia

Proyecto académico desarrollado en el marco del Máster en IA y Big Data. Uso educativo.
