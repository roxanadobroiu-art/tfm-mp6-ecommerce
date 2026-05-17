"""
Modelo Prophet para prediccion de ventas diarias.

Proyecto: MP6 - IA y Big Data
Modelo asignado: Prophet (FBProphet) - modelo estadistico de Facebook
                 disenado para capturar tendencias y estacionalidades.

Dataset de entrada: dataset_original_limpio_es.csv (generado por Limpieza_final.py)
Variable objetivo: importe total de ventas por dia (suma de importe_linea)

Splits (definidos por el profesor):
  - Train/Validation: 1 dic 2010 -> 8 nov 2011
  - Test: 9 nov 2011 -> 9 dic 2011

Metrica principal: RMSE (Root Mean Squared Error)
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACION
# ==============================================================================

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(RUTA_BASE, "dataset_original_limpio_es.csv")
RUTA_GRAFICOS = os.path.join(RUTA_BASE, "graficos_prophet")

FECHA_CORTE_TRAIN = "2011-11-08"  # Ultimo dia de entrenamiento
FECHA_INICIO_TEST = "2011-11-09"  # Primer dia de test
FECHA_FIN_TEST = "2011-12-09"     # Ultimo dia de test

os.makedirs(RUTA_GRAFICOS, exist_ok=True)

# ==============================================================================
# 1. CARGA Y PREPARACION DE DATOS
# ==============================================================================

print("=" * 60)
print("1. CARGA Y PREPARACION DE DATOS")
print("=" * 60)

df = pd.read_csv(RUTA_DATASET, parse_dates=["fecha"])
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Rango de fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}")

# Filtrar outliers y no-productos para la agregacion de ventas diarias.
# Los outliers se marcaron en la limpieza pero no se eliminaron;
# aqui los excluimos para que no distorsionen las predicciones del modelo.
df_ventas = df[
    (~df["flag_outlier_importe"]) &
    (~df["flag_no_producto"])
].copy()

print(f"Filas tras excluir outliers de importe y no-productos: {len(df_ventas)}")

# Agregar ventas totales por dia.
# Prophet requiere un DataFrame con dos columnas: 'ds' (fecha) e 'y' (valor).
ventas_diarias = (
    df_ventas
    .groupby("fecha")["importe_linea"]
    .sum()
    .reset_index()
)
ventas_diarias.columns = ["ds", "y"]
ventas_diarias = ventas_diarias.sort_values("ds").reset_index(drop=True)

print(f"Serie temporal: {len(ventas_diarias)} dias con ventas registradas")
print(f"Ventas diarias - media: {ventas_diarias['y'].mean():,.2f}, "
      f"std: {ventas_diarias['y'].std():,.2f}")

# ==============================================================================
# 2. DIVISION TRAIN / TEST
# ==============================================================================

print("\n" + "=" * 60)
print("2. DIVISION TRAIN / TEST")
print("=" * 60)

train = ventas_diarias[ventas_diarias["ds"] <= FECHA_CORTE_TRAIN].copy()
test = ventas_diarias[
    (ventas_diarias["ds"] >= FECHA_INICIO_TEST) &
    (ventas_diarias["ds"] <= FECHA_FIN_TEST)
].copy()

print(f"Train: {len(train)} dias ({train['ds'].min().date()} a {train['ds'].max().date()})")
print(f"Test:  {len(test)} dias ({test['ds'].min().date()} a {test['ds'].max().date()})")

# ==============================================================================
# 3. ENTRENAMIENTO DEL MODELO PROPHET
# ==============================================================================

print("\n" + "=" * 60)
print("3. ENTRENAMIENTO DEL MODELO PROPHET")
print("=" * 60)

# Configuracion de Prophet:
#   - yearly_seasonality: True -> captura estacionalidad anual
#     (aunque solo tenemos ~1 anio, Prophet puede inferir patrones)
#   - weekly_seasonality: True -> captura patrones semanales
#     (dias laborables vs fin de semana)
#   - daily_seasonality: False -> no aplica, nuestros datos son diarios
#   - changepoint_prior_scale: 0.05 -> controla la flexibilidad de la tendencia
#     (valores mas altos = tendencia mas flexible, riesgo de overfitting)
#   - seasonality_prior_scale: 10 -> controla la fuerza de la estacionalidad
#   - holidays_prior_scale: 10 -> controla el efecto de los festivos

modelo = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10,
    interval_width=0.95,
)

# Anadir estacionalidad mensual como componente adicional.
# Esto puede ayudar a capturar patrones como cierres de mes,
# picos de cobro de nominas, etc.
modelo.add_seasonality(name="mensual", period=30.5, fourier_order=5)

print("Entrenando modelo Prophet...")
modelo.fit(train)
print("Modelo entrenado.")

# ==============================================================================
# 4. PREDICCION
# ==============================================================================

print("\n" + "=" * 60)
print("4. PREDICCION")
print("=" * 60)

# Crear dataframe de fechas futuras para el periodo de test.
# Usamos make_future_dataframe para cubrir todo el rango,
# y luego filtramos solo el periodo de test.
futuro = modelo.make_future_dataframe(
    periods=(pd.Timestamp(FECHA_FIN_TEST) - train["ds"].max()).days,
    freq="D"
)
prediccion = modelo.predict(futuro)

# Filtrar solo las predicciones del periodo de test
pred_test = prediccion[
    (prediccion["ds"] >= FECHA_INICIO_TEST) &
    (prediccion["ds"] <= FECHA_FIN_TEST)
].copy()

# Unir predicciones con valores reales
resultados = test.merge(
    pred_test[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    on="ds",
    how="inner"
)

print(f"Predicciones generadas: {len(pred_test)} dias")
print(f"Dias con dato real y prediccion: {len(resultados)}")

# ==============================================================================
# 5. EVALUACION DEL MODELO
# ==============================================================================

print("\n" + "=" * 60)
print("5. EVALUACION DEL MODELO")
print("=" * 60)

y_real = resultados["y"].values
y_pred = resultados["yhat"].values

# RMSE - metrica principal pedida por el profesor
rmse = np.sqrt(mean_squared_error(y_real, y_pred))

# MAE - error absoluto medio, mas interpretable
mae = mean_absolute_error(y_real, y_pred)

# MAPE - error porcentual medio, para comparar entre modelos con distintas escalas
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

# R2 - coeficiente de determinacion
r2 = r2_score(y_real, y_pred)

print(f"RMSE:  {rmse:,.2f}")
print(f"MAE:   {mae:,.2f}")
print(f"MAPE:  {mape:.2f}%")
print(f"R2:    {r2:.4f}")
print(f"\nMedia ventas test:  {np.mean(y_real):,.2f}")
print(f"Media prediccion:   {np.mean(y_pred):,.2f}")
print(f"RMSE / Media real:  {rmse / np.mean(y_real) * 100:.2f}%")

# ==============================================================================
# 6. GRAFICOS
# ==============================================================================

print("\n" + "=" * 60)
print("6. GENERACION DE GRAFICOS")
print("=" * 60)

# --- Grafico 1: Serie temporal completa con prediccion ---
fig1, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(train["ds"], train["y"], color="steelblue", linewidth=0.8,
         label="Train (datos reales)", alpha=0.7)
ax1.plot(resultados["ds"], resultados["y"], color="green", linewidth=1.5,
         label="Test (datos reales)", marker="o", markersize=4)
ax1.plot(resultados["ds"], resultados["yhat"], color="red", linewidth=1.5,
         label="Test (prediccion Prophet)", marker="s", markersize=4, linestyle="--")
ax1.fill_between(
    resultados["ds"], resultados["yhat_lower"], resultados["yhat_upper"],
    color="red", alpha=0.1, label="Intervalo de confianza 95%"
)
ax1.axvline(x=pd.Timestamp(FECHA_INICIO_TEST), color="gray",
            linestyle=":", linewidth=1, label="Inicio periodo test")
ax1.set_title("Prediccion de ventas diarias con Prophet", fontsize=14)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Ventas diarias (importe)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "01_serie_completa_prediccion.png"), dpi=150)
plt.close()
print("  [OK] 01_serie_completa_prediccion.png")

# --- Grafico 2: Zoom en el periodo de test ---
fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(resultados["ds"], resultados["y"], color="green", linewidth=1.5,
         marker="o", markersize=5, label="Valor real")
ax2.plot(resultados["ds"], resultados["yhat"], color="red", linewidth=1.5,
         marker="s", markersize=5, linestyle="--", label="Prediccion Prophet")
ax2.fill_between(
    resultados["ds"], resultados["yhat_lower"], resultados["yhat_upper"],
    color="red", alpha=0.15, label="Intervalo 95%"
)

# Anotar el RMSE en el grafico
ax2.text(0.02, 0.95, f"RMSE: {rmse:,.2f}\nMAE: {mae:,.2f}\nMAPE: {mape:.1f}%",
         transform=ax2.transAxes, fontsize=10, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

ax2.set_title("Detalle del periodo de test: Real vs Prediccion", fontsize=14)
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Ventas diarias (importe)")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "02_zoom_test_prediccion.png"), dpi=150)
plt.close()
print("  [OK] 02_zoom_test_prediccion.png")

# --- Grafico 3: Componentes del modelo (tendencia + estacionalidades) ---
fig3 = modelo.plot_components(prediccion)
fig3.savefig(os.path.join(RUTA_GRAFICOS, "03_componentes_prophet.png"), dpi=150)
plt.close()
print("  [OK] 03_componentes_prophet.png")

# --- Grafico 4: Errores por dia ---
fig4, ax4 = plt.subplots(figsize=(12, 5))

errores = resultados["y"] - resultados["yhat"]
colores = ["green" if e >= 0 else "red" for e in errores]
ax4.bar(resultados["ds"], errores, color=colores, alpha=0.7, width=0.8)
ax4.axhline(y=0, color="black", linewidth=0.5)
ax4.set_title("Error de prediccion por dia (Real - Prediccion)", fontsize=14)
ax4.set_xlabel("Fecha")
ax4.set_ylabel("Error (positivo = infraestimacion)")
ax4.grid(True, alpha=0.3, axis="y")
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "04_errores_por_dia.png"), dpi=150)
plt.close()
print("  [OK] 04_errores_por_dia.png")

# --- Grafico 5: Dispersion real vs predicho ---
fig5, ax5 = plt.subplots(figsize=(7, 7))

ax5.scatter(y_real, y_pred, color="steelblue", alpha=0.7, edgecolors="navy", s=50)
lim_min = min(y_real.min(), y_pred.min()) * 0.9
lim_max = max(y_real.max(), y_pred.max()) * 1.1
ax5.plot([lim_min, lim_max], [lim_min, lim_max], color="red",
         linestyle="--", linewidth=1, label="Prediccion perfecta")
ax5.set_xlim(lim_min, lim_max)
ax5.set_ylim(lim_min, lim_max)
ax5.set_title("Dispersion: Valor real vs Prediccion", fontsize=14)
ax5.set_xlabel("Ventas reales")
ax5.set_ylabel("Prediccion Prophet")
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, "05_dispersion_real_vs_pred.png"), dpi=150)
plt.close()
print("  [OK] 05_dispersion_real_vs_pred.png")

# ==============================================================================
# 7. TABLA RESUMEN DE RESULTADOS
# ==============================================================================

print("\n" + "=" * 60)
print("7. TABLA RESUMEN DIA A DIA")
print("=" * 60)

tabla = resultados[["ds", "y", "yhat"]].copy()
tabla["error"] = tabla["y"] - tabla["yhat"]
tabla["error_pct"] = (tabla["error"].abs() / tabla["y"] * 100).round(1)
tabla["ds"] = tabla["ds"].dt.strftime("%Y-%m-%d")
tabla.columns = ["Fecha", "Real", "Prediccion", "Error", "Error %"]
print(tabla.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# Exportar resultados a CSV
tabla.to_csv(os.path.join(RUTA_BASE, "resultados_prophet.csv"), index=False)
print(f"\nResultados exportados a: resultados_prophet.csv")

# ==============================================================================
# 8. RESUMEN FINAL
# ==============================================================================

print("\n" + "=" * 60)
print("8. RESUMEN FINAL DEL MODELO PROPHET")
print("=" * 60)
print(f"  Modelo: Prophet (Facebook / Meta)")
print(f"  Periodo de entrenamiento: {train['ds'].min().date()} a {train['ds'].max().date()}")
print(f"  Periodo de test: {FECHA_INICIO_TEST} a {FECHA_FIN_TEST}")
print(f"  Dias de entrenamiento: {len(train)}")
print(f"  Dias de test: {len(resultados)}")
print(f"  ---")
print(f"  RMSE:  {rmse:,.2f}")
print(f"  MAE:   {mae:,.2f}")
print(f"  MAPE:  {mape:.2f}%")
print(f"  R2:    {r2:.4f}")
print(f"  ---")
print(f"  Graficos guardados en: {RUTA_GRAFICOS}/")
print("=" * 60)
