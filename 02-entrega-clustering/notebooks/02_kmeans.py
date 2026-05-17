import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RUTA_ENTRADA = r"C:\Users\ruizp\Documents\Estudios\master\Asignaturas\TFM\Clustering\dataset_clientes_clustering.csv"
RUTA_SALIDA_CLIENTES = r"C:\Users\ruizp\Documents\New project\clientes_kmeans_resultado.csv"
def preparar_dataset(df):
    df = df.copy()

    df["primer_compra"] = pd.to_datetime(df["primer_compra"], errors="coerce")
    df["ultima_compra"] = pd.to_datetime(df["ultima_compra"], errors="coerce")

    fecha_referencia = df["ultima_compra"].max()
    df["recencia_dias"] = (fecha_referencia - df["ultima_compra"]).dt.days

    columnas_log = [
        "total_compras",
        "total_productos_diferentes",
        "total_facturas",
        "valor_total_compras",
        "valor_medio_por_compra",
        "valor_mediano_por_compra",
        "precio_unitario_medio",
        "cantidad_total_productos",
        "cantidad_media_por_compra",
        "dias_como_cliente",
        "compras_fines_semana",
        "tiempo_medio_entre_compras",
        "tiempo_mediano_entre_compras",
        "frecuencia_compra",
        "meses_activo",
        "valor_mensual_medio",
        "compras_mensuales_medias",
        "recencia_dias",
    ]

    for columna in columnas_log:
        df[f"{columna}_log"] = np.log1p(df[columna].clip(lower=0))

    return df

# ============================
# 1. CARGA Y PREPARACIÓN
# ============================
df = pd.read_csv(RUTA_ENTRADA)
df = preparar_dataset(df)

columnas_numericas = [
    "total_compras_log",
    "total_productos_diferentes_log",
    "total_facturas_log",
    "valor_total_compras_log",
    "valor_medio_por_compra_log",
    "precio_unitario_medio_log",
    "cantidad_total_productos_log",
    "cantidad_media_por_compra_log",
    "dias_como_cliente_log",
    "paises_diferentes",
    "categorias_productos_diferentes",
    "compras_fines_semana_log",
    "porcentaje_compras_fines_semana",
    "mes_mas_activo",
    "meses_diferentes_compra",
    "tiempo_medio_entre_compras_log",
    "frecuencia_compra_log",
    "meses_activo_log",
    "valor_mensual_medio_log",
    "compras_mensuales_medias_log",
    "diversidad_categorias",
    "recencia_dias_log",
]

columnas_categoricas = [
    "pais_principal",
    "categoria_favorita",
    "dia_semana_favorito",
    "hora_favorita",
]

# ============================
# 2. PREPROCESADOR
# ============================
preprocesador = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), columnas_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
    ]
)

X = preprocesador.fit_transform(df)

# ============================
# 3. PCA (CLAVE PARA SUBIR SILHOUETTE)
# ============================
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)

print("Varianza explicada PCA:", pca.explained_variance_ratio_.sum())

# ============================
# 4. OUTLIERS (LOF)
# ============================
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
labels_lof = lof.fit_predict(X_pca)

mask = labels_lof == 1
X_modelo = X_pca[mask]
df_modelo = df[mask].copy()

print("Clientes tras LOF:", len(df_modelo))

# ============================
# 5. BUSQUEDA DE K
# ============================
resultados = []

for k in range(2, 10):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=30)
    etiquetas = modelo.fit_predict(X_modelo)
    sil = silhouette_score(X_modelo, etiquetas)
    resultados.append((k, sil))

resultados = sorted(resultados, key=lambda x: x[1], reverse=True)
print("Resultados silhouette:", resultados)

mejor_k = resultados[0][0]
print("Mejor k:", mejor_k)

# ============================
# 6. MODELO FINAL
# ============================
modelo_final = KMeans(n_clusters=mejor_k, random_state=42, n_init=50)
df_modelo["cluster"] = modelo_final.fit_predict(X_modelo)

# ============================
# 7. EXPORTACIÓN
# ============================
df_modelo.to_csv(RUTA_SALIDA_CLIENTES, index=False, encoding="utf-8")
print("Exportado:", RUTA_SALIDA_CLIENTES)
