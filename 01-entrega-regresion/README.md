# Entrega 1 — Regresión: Predicción de ventas diarias

Proyecto de regresión sobre series temporales para predecir las ventas diarias del *e-commerce*.

---

## 📌 Objetivo

Predecir el importe total de ventas por día (suma de `importe_linea`) usando como entrada el histórico de transacciones del dataset *E-Commerce Data*. La predicción se utiliza para apoyar la planificación de inventario, campañas comerciales y dimensionamiento operativo.

---

## 📊 División temporal

- **Train/Validación:** 1 dic 2010 → 8 nov 2011
- **Test:** 9 nov 2011 → 9 dic 2011
- **Métrica principal:** RMSE

---

## 🛠️ Pipeline de preprocesado

1. **Carga y renombrado** del dataset original a español.
2. **Limpieza básica:** duplicados, notas de crédito (prefijo `C`), cantidades ≤ 0, filas sin `id_cliente`.
3. **Variables derivadas:** fechas, día de la semana, importe por línea, categoría de producto.
4. **Detección de festivos por país** (con caché ISO).
5. **Detección de outliers** con regla 1,5 IQR sobre cantidad, precio e importe.
6. **Exportación** del dataset limpio.

---

## 🤖 Modelos implementados

| Script | Modelo | Tipo |
|---|---|---|
| `02_modelo_prophet.py` | Prophet | Series temporales descompuestas |
| `03_modelo_arima.py` | ARIMA(1,1,1) | Series temporales clásicas |
| `04_modelo_random_forest.py` | Random Forest | *Ensemble* de árboles |
| `05_modelo_xgboost.py` | XGBoost | *Gradient Boosting* |
| `06_modelo_lstm.py` | LSTM | Red neuronal recurrente |
| `07_regresion_polinomica.py` | Regresión Polinómica | Modelo lineal con términos polinómicos |

---

## 📁 Contenido de esta carpeta

```
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
    └── M-IA_C088_Entrega_Intermaedia1_GrupoH1.pdf
```

---

## 🚀 Cómo ejecutar

1. Descomprime `datos/data.csv.zip` para obtener `data.csv`.
2. Ejecuta primero `01_limpieza_dataset.py` para generar el dataset preprocesado.
3. Cada modelo se puede ejecutar de forma independiente con `python NN_modelo_xxx.py`.

> Nota: los scripts contienen rutas absolutas locales que deberán adaptarse al entorno de ejecución.

---

## 📖 Memoria

La memoria completa de esta entrega se encuentra en `memoria/`. Documenta el pipeline de preprocesado, la configuración de cada modelo, los resultados sobre validación y test, el análisis visual y las consideraciones de validez.

---

## 🔗 Volver al README principal

[← README principal del repositorio](../README.md)
