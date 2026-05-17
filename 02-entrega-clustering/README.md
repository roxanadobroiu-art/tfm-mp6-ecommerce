# Entrega 2 — Clustering: Segmentación de clientes

Proyecto de aprendizaje no supervisado para segmentar la base de clientes del *e-commerce* y permitir acciones de *marketing* personalizadas.

---

## 📌 Objetivo

Identificar grupos homogéneos de clientes con comportamientos de compra similares para personalizar estrategias de *marketing*, promociones y comunicación. A diferencia del proyecto de regresión, es un problema **no supervisado**: no existe variable objetivo y no hay división *train* / *test*.

---

## 📊 Dataset de partida

- **Fuente:** dataset de clientes generado a partir del proyecto de regresión, agregando las transacciones por `id_cliente`.
- **Tamaño:** 4.339 clientes únicos.
- **Variables:** 30 (volumen de compras, valor monetario, antigüedad, frecuencia, geografía, estacionalidad, diversidad de productos).

El preprocesado común que genera este dataset se encuentra en `notebooks/01_limpieza_clustering.py`. A partir de ahí, **cada modelo aplica su propio preprocesado específico** (escalado, codificación, eliminación de *outliers*, reducción de dimensionalidad) en función de las particularidades del algoritmo.

---

## 🤖 Modelos implementados

| Script | Modelo | *Clusters* | Silueta |
|---|---|---|---|
| `02_kmeans.py` ⭐ | K-Means | 2 | **0,5257** |
| `03_clustering_jerarquico.py` | Clustering Jerárquico | 2 | 0,7403 |
| `04_dbscan.py` | DBSCAN | 4 | 0,3386 |
| `05_hdbscan.py` | HDBSCAN | 8 | 0,248 |

---

## ⭐ Modelo seleccionado: K-Means

Tras analizar los cuatro modelos, el grupo selecciona **K-Means** como modelo principal por:

- Las tres métricas internas (silueta, Calinski-Harabasz, Davies-Bouldin) **convergen unánimemente en k=2**.
- Silueta de 0,5257 con caída pronunciada a 0,2627 en k=3.
- Preprocesado mejorado con **PCA (10 componentes, 91,8 % de varianza) + LOF** para *outliers*.
- Conserva el 97 % del dataset original (4.208 sobre 4.339).
- Simplicidad, rapidez e interpretabilidad de negocio.
- Aplicabilidad directa en CRM mediante exportación a CSV.

**El Clustering Jerárquico** queda como segunda opción válida: aunque obtiene mejor silueta (0,7403), su mayor coste computacional y la coincidencia con K-Means en estructura (k=2, tamaños ~12 % / ~88 %) hacen que K-Means sea preferible. La coincidencia entre dos algoritmos distintos **valida la estructura** encontrada como propiedad real del dataset.

**DBSCAN y HDBSCAN se descartan** por su elevado porcentaje de ruido (49,3 % y 26,0 % respectivamente) y por la presencia de *clusters* minúsculos.

---

## 📁 Contenido de esta carpeta

```
02-entrega-clustering/
├── datos/
│   └── dataset_clientes_clustering.csv    # Dataset preprocesado de clientes
├── notebooks/
│   ├── 01_limpieza_clustering.py          # Preprocesado común
│   ├── 02_kmeans.py                       # Modelo principal
│   ├── 03_clustering_jerarquico.py
│   ├── 04_dbscan.py
│   └── 05_hdbscan.py
└── memoria/
    ├── memoria_clustering.pdf
    └── memoria_clustering.tex
```

---

## 🚀 Cómo ejecutar

1. El dataset preprocesado ya está disponible en `datos/dataset_clientes_clustering.csv`.
2. Cada modelo se puede ejecutar de forma independiente con `python NN_modelo.py`.
3. El script `01_limpieza_clustering.py` permite regenerar el dataset de clientes desde el dataset transaccional original (en `01-entrega-regresion/datos/`).

> Nota: los scripts contienen rutas absolutas locales que deberán adaptarse al entorno de ejecución.

---

## 📖 Memoria

La memoria completa (35 páginas) se encuentra en `memoria/`. Documenta el preprocesado común, los cuatro modelos con su preprocesado específico, los resultados, análisis visual, interpretación de *clusters*, y la conclusión con el modelo seleccionado.

---

## 🔗 Volver al README principal

[← README principal del repositorio](../README.md)
