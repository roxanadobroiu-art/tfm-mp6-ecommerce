# Entrega 1 — Regresión: Predicción de ventas diarias

Proyecto de regresión sobre series temporales para predecir las ventas diarias del *e-commerce*.

---

## 📌 Objetivo

Predecir el importe total de ventas por día (suma de `importe_linea`) usando como entrada el histórico de transacciones del dataset *E-Commerce Data*. La predicción se utiliza para apoyar la planificación de inventario, campañas comerciales y dimensionamiento operativo.

---

## 📊 División temporal

| Conjunto | Periodo | Días con ventas |
|---|---|---|
| Train | 1 dic 2010 → 8 oct 2011 | 251 |
| Validación | 9 oct 2011 → 8 nov 2011 | 27 |
| Test | 9 nov 2011 → 9 dic 2011 | 27 |

**Métrica principal:** RMSE (complementada con MAE, MAPE y R²).

---

## 🛠️ Pipeline de preprocesado

1. **Carga y renombrado** del dataset original a español.
2. **Limpieza básica:** duplicados, notas de crédito (prefijo `C`), cantidades ≤ 0, filas sin `id_cliente`, precios no válidos.
3. **Variables derivadas:** fechas, día de la semana, importe por línea, categoría de producto.
4. **Detección de festivos por país** (con caché ISO).
5. **Detección de outliers** con regla 1,5·IQR sobre cantidad, precio e importe (marcados con *flags*, no eliminados).
6. **Regla de negocio:** se conservan los *outliers* que coinciden con festivos.
7. **Exportación** del dataset limpio (361.126 filas marcadas para entrenamiento).

---

## 🤖 Modelos implementados

| Script | Modelo | Tipo |
|---|---|---|
| `02_modelo_prophet.py` | Prophet | Series temporales descompuestas |
| `03_modelo_arima.py` | ARIMA(1,1,1) | Series temporales clásicas |
| `04_modelo_random_forest.py` | Random Forest | *Ensemble* de árboles (*bagging*) |
| `05_modelo_xgboost.py` | XGBoost | *Gradient Boosting* |
| `06_modelo_lstm.py` | LSTM | Red neuronal recurrente |
| `07_regresion_polinomica.py` | Regresión Polinómica | Modelo lineal con términos polinómicos |

---

## 🏆 Resultados (conjunto de test)

| Modelo | RMSE (GBP) | R² | MAE (GBP) |
|---|---|---|---|
| **XGBoost** ⭐ | **19.604** | **0,56** | **9.198** |
| Random Forest | 27.402 | 0,14 | 13.083 |
| ARIMA(1,1,1) | 31.688 | — | — |
| LSTM | 35.401 | −0,19 | 24.007 |
| Prophet | 35.789 | — | — |
| Regresión Polinómica\* | 17.058 | −0,43 | 14.836 |

⭐ **Modelo seleccionado: XGBoost.** Mejor RMSE y R² del conjunto, con 23 *features* construidas manualmente. Supera al *baseline naïve* en un +38,8 %.

\* *El RMSE de la Regresión Polinómica es engañosamente bajo: su R² negativo indica que rinde peor que predecir la media, y su partición train/test es distinta a la del resto, por lo que no entra en el ranking competitivo.*

---

## 📁 Contenido de esta carpeta
01-entrega-regresion/
├── datos/
│   └── data.csv.zip                       # Dataset original (comprimido)
├── notebooks/
│   ├── 01_limpieza_dataset.py             # Pipeline de limpieza
│   ├── 02_modelo_prophet.py
│   ├── 03_modelo_arima.py
│   ├── 04_modelo_random_forest.py
│   ├── 05_modelo_xgboost.py
│   ├── 06_modelo_lstm.py
│   └── 07_regresion_polinomica.py
└── memoria/
└── (memoria de la entrega en PDF)


---

## 🚀 Cómo ejecutar

1. Descomprime `datos/data.csv.zip` para obtener `data.csv`.
2. Ejecuta primero `01_limpieza_dataset.py` para generar el dataset preprocesado.
3. Cada modelo se puede ejecutar de forma independiente con `python NN_modelo_xxx.py`.

> Nota: los scripts contienen rutas absolutas locales que deberán adaptarse al entorno de ejecución.

---

## 📖 Memoria

La memoria completa de esta entrega se encuentra en `memoria/`. Documenta el pipeline de preprocesado, la configuración de cada modelo, los resultados sobre validación y test, el análisis visual y las consideraciones de validez de cada modelo, además de la comparativa final y la justificación de la elección de XGBoost.

---

## 🔗 Volver al README principal

[← README principal del repositorio](../README.md)

