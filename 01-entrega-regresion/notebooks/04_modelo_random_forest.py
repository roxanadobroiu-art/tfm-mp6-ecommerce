"""
Random Forest para predicción de series temporales de ventas.
Objetivo: predecir ventas_diarias = suma(importe_linea por fecha).
Fecha de corte: 2011-11-09
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FECHA_CORTE = '2011-11-09'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Carga el dataset transaccional y convierte la columna de fecha al tipo datetime
def cargar_dataset():
    df = pd.read_csv("DataSet/dataset_clusterings_base.csv")
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    return df


# Agrega los datos a nivel diario: una fila por día con métricas operacionales
def agregar_datos_diarios(df):
    df_daily = df.groupby('fecha').agg({
        'importe_linea': 'sum',
        'cantidad': 'sum',
        'numero_factura': 'nunique',
        'id_cliente': 'nunique',
        'codigo_producto': 'nunique'
    }).reset_index()

    df_daily.columns = ['fecha', 'ventas_diarias', 'cantidad_total',
                        'num_facturas', 'clientes_unicos', 'productos_unicos']
    return df_daily


# Construye todas las features de series temporales evitando data leakage:
# todos los lags y rolling windows usan shift(1) para garantizar que en el
# instante t el modelo solo accede a información disponible hasta t-1
def crear_features_series_temporales(df_daily):
    df = df_daily.copy().sort_values('fecha').reset_index(drop=True)

    # Lags de la variable objetivo a distintas escalas temporales
    df['ventas_lag_1'] = df['ventas_diarias'].shift(1)
    df['ventas_lag_2'] = df['ventas_diarias'].shift(2)
    df['ventas_lag_3'] = df['ventas_diarias'].shift(3)
    df['ventas_lag_7'] = df['ventas_diarias'].shift(7)
    df['ventas_lag_14'] = df['ventas_diarias'].shift(14)
    df['ventas_lag_30'] = df['ventas_diarias'].shift(30)

    # Medias y desviación típica móviles sobre el pasado inmediato
    df['rolling_mean_3'] = df['ventas_diarias'].shift(1).rolling(window=3, min_periods=1).mean()
    df['rolling_mean_7'] = df['ventas_diarias'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_mean_14'] = df['ventas_diarias'].shift(1).rolling(window=14, min_periods=1).mean()
    df['rolling_std_7'] = df['ventas_diarias'].shift(1).rolling(window=7, min_periods=1).std()

    # Features de calendario para capturar estacionalidades sistemáticas
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['mes'] = df['fecha'].dt.month
    df['es_fin_de_semana'] = (df['fecha'].dt.dayofweek >= 5).astype(int)
    df['dia_mes'] = df['fecha'].dt.day
    df['es_inicio_mes'] = (df['fecha'].dt.day <= 5).astype(int)
    df['es_final_mes'] = (df['fecha'].dt.day >= 25).astype(int)

    # Diferencias temporales para capturar cambios bruscos
    df['diff_1d'] = df['ventas_diarias'].diff(1)
    df['diff_7d'] = df['ventas_diarias'].diff(7)

    # Ratios que miden la posición actual respecto al pasado reciente
    df['ratio_vs_lag7'] = df['ventas_lag_1'] / (df['ventas_lag_7'] + 1e-6)
    df['ratio_vs_mean7'] = df['ventas_lag_1'] / (df['rolling_mean_7'] + 1e-6)

    # Lags de métricas operacionales del día anterior
    df['clientes_lag_1'] = df['clientes_unicos'].shift(1)
    df['facturas_lag_1'] = df['num_facturas'].shift(1)
    df['productos_lag_1'] = df['productos_unicos'].shift(1)

    features_finales = [
        'ventas_lag_1', 'ventas_lag_2', 'ventas_lag_3', 'ventas_lag_7', 'ventas_lag_14', 'ventas_lag_30',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
        'dia_semana', 'mes', 'es_fin_de_semana', 'dia_mes', 'es_inicio_mes', 'es_final_mes',
        'diff_1d', 'diff_7d',
        'ratio_vs_lag7', 'ratio_vs_mean7',
        'clientes_lag_1', 'facturas_lag_1', 'productos_lag_1'
    ]

    return df, features_finales


# Elimina las filas con NaN generadas por los shift() de los lags
def limpiar_tras_lags(df, features_finales):
    df_clean = df.dropna(subset=features_finales + ['ventas_diarias'])
    return df_clean


# Realiza el split temporal estricto: train antes de la fecha de corte,
# test a partir de ella. Aplica log1p al target para estabilizar la varianza
def split_temporal(df_clean, features_finales):
    fecha_corte = pd.to_datetime(FECHA_CORTE)

    train_data = df_clean[df_clean['fecha'] < fecha_corte].copy()
    test_data = df_clean[df_clean['fecha'] >= fecha_corte].copy()

    X_train = train_data[features_finales]
    y_train = train_data['ventas_diarias']
    X_test = test_data[features_finales]
    y_test = test_data['ventas_diarias']

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    return X_train, X_test, y_train_log, y_test_log, y_train, y_test, train_data, test_data


# Entrena el modelo Random Forest con hiperparámetros ajustados para series temporales
def entrenar_random_forest(X_train, y_train_log):
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=5,
        max_features='sqrt',
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True
    )
    model.fit(X_train, y_train_log)
    return model


# Valida el modelo con TimeSeriesSplit para respetar el orden cronológico de los datos
def validacion_temporal(X_train, y_train_log):
    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores, rmse_scores, mape_scores = [], [], []

    for train_idx, val_idx in tscv.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

        model_cv = RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_cv.fit(X_fold_train, y_fold_train)

        y_pred = np.expm1(model_cv.predict(X_fold_val))
        y_val_orig = np.expm1(y_fold_val)

        r2_scores.append(r2_score(y_val_orig, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_orig, y_pred)))
        mape_scores.append(np.mean(np.abs((y_val_orig - y_pred) / y_val_orig)) * 100)

    cv_stats = {
        'R2_mean': np.mean(r2_scores), 'R2_std': np.std(r2_scores),
        'RMSE_mean': np.mean(rmse_scores), 'RMSE_std': np.std(rmse_scores),
        'MAPE_mean': np.mean(mape_scores), 'MAPE_std': np.std(mape_scores)
    }
    return cv_stats


# Genera predicciones sobre el test set y calcula las métricas en escala original
def evaluar_modelo_final(model, X_test, y_test_log, y_test, test_data):
    y_pred = np.expm1(model.predict(X_test))

    metricas = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    return y_pred, metricas


# Evalúa el baseline naive: predecir las ventas de hoy como las del día anterior
def baseline_naive(test_data, y_test):
    y_baseline = test_data['ventas_lag_1'].values

    metricas_baseline = {
        'MAE': mean_absolute_error(y_test, y_baseline),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_baseline)),
        'R2': r2_score(y_test, y_baseline),
        'MAPE': np.mean(np.abs((y_test - y_baseline) / y_test)) * 100
    }
    return metricas_baseline


# Extrae y ordena la importancia de cada feature según el modelo entrenado
def analizar_importancia_features(model, features_finales):
    importancias = pd.DataFrame({
        'feature': features_finales,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return importancias


# Imprime la comparación de métricas entre el modelo y el baseline naive
def comparar_con_baseline(metricas_modelo, metricas_baseline):
    print(f"\n{'Métrica':<12} {'Modelo':<12} {'Baseline':<12} {'Mejora':<12}")
    print("-" * 50)

    for metrica in ['R2', 'MAE', 'RMSE', 'MAPE']:
        modelo_val = metricas_modelo[metrica]
        baseline_val = metricas_baseline[metrica]

        if metrica == 'R2':
            mejora = ((modelo_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
        else:
            mejora = ((baseline_val - modelo_val) / baseline_val) * 100 if baseline_val != 0 else 0

        mejora_str = f"+{mejora:.1f}%" if mejora > 0 else f"{mejora:.1f}%"
        print(f"{metrica:<12} {modelo_val:<12.4f} {baseline_val:<12.4f} {mejora_str:<12}")


# Genera y guarda el gráfico de comparación predicción vs realidad como PNG
def guardar_grafico_prediccion_vs_real(test_data, y_test, y_pred, nombre_archivo='prediccion_vs_realidad.png'):
    # Importaciones necesarias para las métricas
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # PALETA DE COLORES AMIGABLE PARA DALTÓNICOS
    colores_daltonicos = {
        'real': '#2ca02c',         # Verde
        'prediccion': '#ff7f0e',   # Naranja
    }
    
    plt.figure(figsize=(14, 8))
    
    fechas_test = test_data['fecha'].values
    
    # Calcular métricas para mostrar
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Gráficos con marcadores para mejor distinción
    plt.plot(fechas_test, y_test, label=f'Ventas Reales (μ=${np.mean(y_test):,.0f})', 
            linewidth=3, alpha=0.8, color=colores_daltonicos['real'], marker='o', markersize=5)
    plt.plot(fechas_test, y_pred, label=f'Predicción (MAE=${mae:,.0f}, R²={r2:.3f})', 
            linewidth=3, alpha=0.8, color=colores_daltonicos['prediccion'], marker='s', markersize=5)
    
    # Añadir anotaciones con valores clave
    max_real = np.max(y_test)
    min_real = np.min(y_test)
    max_idx = np.argmax(y_test)
    min_idx = np.argmin(y_test)
    
    plt.annotate(f'Máx Real: ${max_real:,.0f}', 
                xy=(fechas_test[max_idx], max_real), 
                xytext=(10, 15), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                fontsize=11, fontweight='bold')
    
    plt.annotate(f'Mín Real: ${min_real:,.0f}', 
                xy=(fechas_test[min_idx], min_real), 
                xytext=(10, -25), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                fontsize=11, fontweight='bold')
    
    plt.title('Comparación: Predicción vs Datos Reales (Optimizado para Daltonismo)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12, fontweight='bold')
    plt.ylabel('Ventas Diarias ($)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar el gráfico como PNG
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado como: {nombre_archivo}")
    
    # Mostrar el gráfico
    plt.show()


# Genera y guarda cuatro gráficos individuales: serie temporal, scatter, importancia de features y residuos
def visualizar_resultados(test_data, y_test, y_pred, importancias, df_clean=None):
    # Importaciones necesarias para las métricas
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    fechas_test = test_data['fecha'].values
    
    # PALETA DE COLORES AMIGABLE PARA DALTÓNICOS
    colores_daltonicos = {
        'historico': '#1f77b4',    # Azul
        'real': '#2ca02c',         # Verde
        'prediccion': '#ff7f0e',   # Naranja
        'corte': '#d62728',        # Rojo
        'scatter_main': '#9467bd', # Morado
        'scatter_line': '#8c564b', # Marrón
        'residuos': '#17becf',     # Cian
        'barras': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    }
    
    # ==============================================================================
    # GRÁFICO 1: SERIE TEMPORAL COMPLETA - CONTEXTO HISTÓRICO + PREDICCIONES
    # ==============================================================================
    plt.figure(figsize=(16, 10))
    
    if df_clean is not None:
        # Mostrar toda la serie temporal histórica
        fecha_corte = pd.to_datetime(FECHA_CORTE)
        
        # Datos de entrenamiento (histórico)
        train_mask = df_clean['fecha'] < fecha_corte
        fechas_train = df_clean[train_mask]['fecha'].values
        ventas_train = df_clean[train_mask]['ventas_diarias'].values
        
        # Estadísticas del período histórico
        media_historica = np.mean(ventas_train)
        std_historica = np.std(ventas_train)
        
        # Graficar serie histórica completa
        plt.plot(fechas_train, ventas_train, label=f'Datos Históricos (μ=${media_historica:,.0f}, σ=${std_historica:,.0f})', 
                linewidth=2, alpha=0.7, color=colores_daltonicos['historico'], linestyle='-')
        
        # Línea vertical para marcar el corte temporal
        plt.axvline(x=fecha_corte, color=colores_daltonicos['corte'], linestyle='--', linewidth=3, 
                   alpha=0.8, label=f'Corte Temporal: {FECHA_CORTE}')
    
    # Estadísticas del período de test
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    
    # Datos de test: reales vs predicciones
    plt.plot(fechas_test, y_test, label=f'Ventas Reales Test (μ=${np.mean(y_test):,.0f})', 
            linewidth=3, alpha=0.9, color=colores_daltonicos['real'], marker='o', markersize=4)
    plt.plot(fechas_test, y_pred, label=f'Predicciones (MAE=${mae_test:,.0f}, R²={r2_test:.3f})', 
            linewidth=3, alpha=0.9, color=colores_daltonicos['prediccion'], marker='s', markersize=4)
    
    # Añadir anotaciones con valores extremos
    max_real_idx = np.argmax(y_test)
    min_real_idx = np.argmin(y_test)
    plt.annotate(f'Máx: ${y_test.iloc[max_real_idx]:,.0f}', 
                xy=(fechas_test[max_real_idx], y_test.iloc[max_real_idx]), 
                xytext=(10, 10), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=10, fontweight='bold')
    plt.annotate(f'Mín: ${y_test.iloc[min_real_idx]:,.0f}', 
                xy=(fechas_test[min_real_idx], y_test.iloc[min_real_idx]), 
                xytext=(10, -20), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.title('Serie Temporal Completa: Contexto Histórico + Predicciones vs Realidad', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Fecha', fontsize=13, fontweight='bold')
    plt.ylabel('Ventas Diarias ($)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('1_serie_temporal_prediccion_vs_realidad.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: 1_serie_temporal_prediccion_vs_realidad.png")
    plt.show()

    # ==============================================================================
    # GRÁFICO 2: SCATTER PLOT - CORRELACIÓN PREDICCIÓN-REALIDAD
    # ==============================================================================
    plt.figure(figsize=(12, 10))
    
    # Calcular métricas para mostrar en el gráfico
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Scatter plot con colores daltónico-amigables
    plt.scatter(y_test, y_pred, alpha=0.7, s=100, color=colores_daltonicos['scatter_main'], 
               edgecolors='black', linewidth=1.5, marker='o')
    
    # Línea de predicción perfecta
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color=colores_daltonicos['scatter_line'], 
            linestyle='--', alpha=0.8, linewidth=3, label='Predicción Perfecta')
    
    # Añadir texto con métricas en el gráfico
    texto_metricas = f'MAE: ${mae:,.0f}\nRMSE: ${rmse:,.0f}\nR²: {r2:.3f}\nMAPE: {mape:.1f}%'
    plt.text(0.05, 0.95, texto_metricas, transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Marcar puntos extremos
    max_error_idx = np.argmax(np.abs(y_test - y_pred))
    plt.scatter(y_test.iloc[max_error_idx], y_pred[max_error_idx], 
               s=200, color='red', marker='x', linewidth=4, 
               label=f'Mayor Error: ${np.abs(y_test.iloc[max_error_idx] - y_pred[max_error_idx]):,.0f}')
    
    plt.xlabel('Ventas Reales ($)', fontsize=13, fontweight='bold')
    plt.ylabel('Ventas Predichas ($)', fontsize=13, fontweight='bold')
    plt.title('Correlación Predicción-Realidad con Métricas de Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('2_correlacion_prediccion_realidad.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: 2_correlacion_prediccion_realidad.png")
    plt.show()

    # ==============================================================================
    # GRÁFICO 3: IMPORTANCIA DE FEATURES (TOP 15)
    # ==============================================================================
    plt.figure(figsize=(14, 12))
    top_features = importancias.head(15)  # Mostrar más features para mejor análisis
    
    # Usar colores daltónico-amigables con gradiente
    colors_gradient = [colores_daltonicos['barras'][i % len(colores_daltonicos['barras'])] 
                      for i in range(len(top_features))]
    
    # Crear barras horizontales
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                   color=colors_gradient, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Añadir valores numéricos en las barras
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        plt.text(width + max(top_features['importance']) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Añadir porcentajes acumulados
    importancia_total = top_features['importance'].sum()
    acumulado = 0
    for i, importance in enumerate(top_features['importance']):
        acumulado += importance
        porcentaje_acum = (acumulado / importancia_total) * 100
        plt.text(max(top_features['importance']) * 0.7, i, 
                f'{porcentaje_acum:.1f}%', ha='center', va='center', 
                fontsize=9, style='italic', alpha=0.7)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia Relativa', fontsize=13, fontweight='bold')
    plt.ylabel('Features', fontsize=13, fontweight='bold')
    plt.title(f'Top 15 Features Más Importantes - Random Forest\n(Importancia Total Mostrada: {importancia_total:.3f})', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('3_importancia_features.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: 3_importancia_features.png")
    plt.show()

    # ==============================================================================
    # GRÁFICO 4: ANÁLISIS DE RESIDUOS
    # ==============================================================================
    plt.figure(figsize=(12, 10))
    residuos = y_test - y_pred
    
    # Estadísticas de residuos para mostrar
    media_residuos = np.mean(residuos)
    std_residuos = np.std(residuos)
    
    # Scatter plot de residuos
    plt.scatter(y_pred, residuos, alpha=0.7, s=100, color=colores_daltonicos['residuos'], 
               edgecolors='black', linewidth=1.5, marker='o')
    
    # Líneas de referencia
    plt.axhline(y=0, color=colores_daltonicos['corte'], linestyle='-', linewidth=3, 
               alpha=0.8, label='Residuo = 0 (Predicción Perfecta)')
    plt.axhline(y=media_residuos, color=colores_daltonicos['prediccion'], linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Media Residuos = ${media_residuos:.0f}')
    
    # Bandas de confianza (±1σ, ±2σ)
    plt.axhline(y=media_residuos + std_residuos, color='gray', linestyle=':', alpha=0.5, label='±1σ')
    plt.axhline(y=media_residuos - std_residuos, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=media_residuos + 2*std_residuos, color='gray', linestyle='-.', alpha=0.5, label='±2σ')
    plt.axhline(y=media_residuos - 2*std_residuos, color='gray', linestyle='-.', alpha=0.5)
    
    # Identificar outliers (residuos > 2σ)
    outliers_mask = np.abs(residuos - media_residuos) > 2 * std_residuos
    if np.any(outliers_mask):
        outliers_pred = y_pred[outliers_mask]
        outliers_res = residuos[outliers_mask]
        plt.scatter(outliers_pred, outliers_res, s=150, color='red', marker='X', 
                   linewidth=3, label=f'Outliers (>2σ): {np.sum(outliers_mask)}')
    
    # Texto con estadísticas
    texto_stats = f'Media: ${media_residuos:.0f}\nStd: ${std_residuos:.0f}\nOutliers: {np.sum(outliers_mask)}/{len(residuos)}'
    plt.text(0.02, 0.98, texto_stats, transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.xlabel('Predicciones ($)', fontsize=13, fontweight='bold')
    plt.ylabel('Residuos ($)', fontsize=13, fontweight='bold')
    plt.title('Análisis de Residuos con Bandas de Confianza', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('4_analisis_residuos.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: 4_analisis_residuos.png")
    plt.show()
    
    # ==============================================================================
    # GRÁFICO 5: DISTRIBUCIÓN DE RESIDUOS (NUEVO)
    # ==============================================================================
    plt.figure(figsize=(12, 10))
    
    # Histograma con colores daltónico-amigables
    n, bins, patches = plt.hist(residuos, bins=20, color=colores_daltonicos['historico'], 
                               edgecolor='black', alpha=0.8, density=True, linewidth=1.5)
    
    # Líneas de referencia
    plt.axvline(x=0, color=colores_daltonicos['corte'], linestyle='--', linewidth=3, 
               alpha=0.8, label='Residuo = 0 (Ideal)')
    plt.axvline(x=media_residuos, color=colores_daltonicos['prediccion'], linestyle='-', 
               linewidth=3, alpha=0.8, label=f'Media = ${media_residuos:.0f}')
    
    # Añadir líneas de desviación estándar
    plt.axvline(x=media_residuos + std_residuos, color='gray', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'+1σ = ${media_residuos + std_residuos:.0f}')
    plt.axvline(x=media_residuos - std_residuos, color='gray', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'-1σ = ${media_residuos - std_residuos:.0f}')
    
    # Estadísticas adicionales
    mediana_residuos = np.median(residuos)
    q25 = np.percentile(residuos, 25)
    q75 = np.percentile(residuos, 75)
    
    # Texto con estadísticas detalladas
    texto_detallado = f'''Estadísticas de Residuos:
• Media: ${media_residuos:.0f}
• Mediana: ${mediana_residuos:.0f}
• Std: ${std_residuos:.0f}
• Q25: ${q25:.0f}
• Q75: ${q75:.0f}
• Rango: ${residuos.max() - residuos.min():.0f}'''
    
    plt.text(0.02, 0.98, texto_detallado, transform=plt.gca().transAxes, 
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
    
    plt.xlabel('Residuos ($)', fontsize=13, fontweight='bold')
    plt.ylabel('Densidad', fontsize=13, fontweight='bold')
    plt.title(f'Distribución de Residuos\nNormalidad: Media=${media_residuos:.0f}, Std=${std_residuos:.0f}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('5_distribucion_residuos.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: 5_distribucion_residuos.png")
    plt.show()

    # ==============================================================================
    # GRÁFICO 6: MÉTRICAS DE PERFORMANCE (NUEVO)
    # ==============================================================================
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    plt.figure(figsize=(14, 8))
    metricas = ['MAE\n(Mean Absolute Error)', 'RMSE\n(Root Mean Squared Error)', 
               'R²\n(Coef. Determinación)', 'MAPE (%)\n(Mean Absolute % Error)']
    valores = [mae, rmse, r2, mape]
    
    # Usar colores específicos para cada métrica
    colors_metricas = [colores_daltonicos['barras'][i] for i in [0, 1, 2, 3]]
    
    bars = plt.bar(metricas, valores, color=colors_metricas, edgecolor='black', 
                   linewidth=2, alpha=0.8, width=0.6)
    
    # Añadir valores numéricos encima de las barras
    for bar, valor, metrica in zip(bars, valores, ['MAE', 'RMSE', 'R²', 'MAPE']):
        height = bar.get_height()
        if metrica in ['MAE', 'RMSE']:
            texto = f'${valor:,.0f}'
        elif metrica == 'R²':
            texto = f'{valor:.4f}'
        else:  # MAPE
            texto = f'{valor:.2f}%'
            
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                texto, ha='center', va='bottom', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Añadir interpretación de las métricas
    interpretacion = f'''Interpretación de Métricas:
• MAE: Error promedio de ${mae:,.0f}
• RMSE: Penaliza errores grandes (${rmse:,.0f})
• R²: Explica {r2*100:.1f}% de la varianza
• MAPE: Error relativo del {mape:.1f}%'''
    
    plt.text(0.02, 0.98, interpretacion, transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.ylabel('Valor de la Métrica', fontsize=13, fontweight='bold')
    plt.title('Métricas de Performance del Modelo Random Forest con Interpretación', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('6_metricas_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: 6_metricas_performance.png")
    plt.show()
    
    print("\n" + "="*80)
    print("✅ TODOS LOS GRÁFICOS INDIVIDUALES GUARDADOS EXITOSAMENTE")
    print("="*80)
    print("📊 Archivos generados:")
    print("   1. prediccion_vs_realidad.png (función separada)")
    print("   2. 1_serie_temporal_prediccion_vs_realidad.png")
    print("   3. 2_correlacion_prediccion_realidad.png")
    print("   4. 3_importancia_features.png")
    print("   5. 4_analisis_residuos.png")
    print("   6. 5_distribucion_residuos.png")
    print("   7. 6_metricas_performance.png")
    print("="*80)


def main():
    df = cargar_dataset()
    df_daily = agregar_datos_diarios(df)
    df_features, features_finales = crear_features_series_temporales(df_daily)
    df_clean = limpiar_tras_lags(df_features, features_finales)
    X_train, X_test, y_train_log, y_test_log, y_train, y_test, train_data, test_data = split_temporal(df_clean, features_finales)
    model = entrenar_random_forest(X_train, y_train_log)
    cv_stats = validacion_temporal(X_train, y_train_log)
    y_pred, metricas_modelo = evaluar_modelo_final(model, X_test, y_test_log, y_test, test_data)
    metricas_baseline = baseline_naive(test_data, y_test)
    importancias = analizar_importancia_features(model, features_finales)
    comparar_con_baseline(metricas_modelo, metricas_baseline)
    
    # Guardar el gráfico individual de predicción vs realidad
    guardar_grafico_prediccion_vs_real(test_data, y_test, y_pred)
    
    # Mostrar todos los gráficos juntos (ahora con contexto histórico completo)
    visualizar_resultados(test_data, y_test, y_pred, importancias, df_clean)

    return model, metricas_modelo, cv_stats, importancias


if __name__ == "__main__":
    model, metricas, cv_stats, importancias = main()
