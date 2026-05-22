# 03 — Entrega Final

Entrega final del proyecto del módulo MP6 del Máster en IA y Big Data.

Contiene la **memoria final unificada** del proyecto de predicción de ventas y segmentación de clientes en e-commerce, que integra las dos entregas anteriores (regresión y clustering) más una parte de conclusiones, interpretación de negocio y reflexión.

---

## 📌 De qué trata el proyecto

A partir del dataset público *E-Commerce Data* (Kaggle / UCI) —transacciones de una tienda británica de regalos y decoración entre dic. 2010 y dic. 2011— se abordan dos problemas complementarios:

- **Regresión:** predecir las ventas diarias (¿cuánto venderemos?).
- **Clustering:** segmentar la base de clientes (¿a quién vendemos?).

---

## 🏆 Resultados principales

| Fase | Modelo elegido | Resultado clave |
|---|---|---|
| Regresión | **XGBoost** | RMSE 19.604 GBP · R² 0,56 · +38,8 % vs. baseline |
| Clustering | **K-Means** | Silueta 0,53 · k=2 (B2B ~88 % / B2C ~12 %) |

- **Regresión:** se comparan seis modelos (Prophet, ARIMA, Regresión Polinómica, Random Forest, XGBoost y LSTM). XGBoost gana en las tres métricas, gracias a 23 *features* construidas manualmente.
- **Clustering:** se comparan cuatro modelos (K-Means, Jerárquico, DBSCAN y HDBSCAN). K-Means y Jerárquico coinciden en k=2, lo que valida la estructura. DBSCAN y HDBSCAN se descartan por exceso de ruido.

---

## 📖 Contenido de la memoria final

La memoria unifica todo el trabajo en tres partes:

1. **Parte I — Regresión:** preprocesado, los seis modelos (configuración, evaluación, análisis visual y validez de cada uno) y comparativa final.
2. **Parte II — Clustering:** preprocesado común, *feature engineering* (marco RFM), los cuatro modelos y comparativa.
3. **Parte III — Conclusiones:** conclusiones de cada fase, recomendaciones de negocio por segmento, diferencias metodológicas, aprendizajes transversales y reflexión del equipo.

---

## 📁 Contenido de esta carpeta

03-entrega-final/
└── memoria/
  └── memoria_final_TFM_GrupoH1.pdf      # Memoria final unificada


---

## 🔗 Entregas anteriores

- [`01-entrega-regresion`](../01-entrega-regresion) — Predicción de ventas diarias (código y notebooks).
- [`02-entrega-clustering`](../02-entrega-clustering) — Segmentación de clientes (código y notebooks).

El código fuente y los datasets de cada fase están en sus carpetas respectivas; esta entrega recoge la documentación final del proyecto.

---

## 👥 Equipo — Grupo H1

Paula Ruiz de la Parra · Elena Roxana Dobroiu · Julián David Dionisio Rey · Javier Santos Rodriguez

---

## 🔗 Volver al README principal

[← README principal del repositorio](../README.md)
