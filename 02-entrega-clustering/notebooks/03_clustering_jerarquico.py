# =============================================================================
# CLUSTERING JERÁRQUICO - Segmentación de clientes E-commerce
# MP6 - Proyecto IA y Big Data | Tema 3
# =============================================================================



# %% [1] IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Librerías cargadas ✓")


# %% [2] CARGA DEL DATASET (ya limpio)
# -----------------------------------------------------------------------------
# En Colab, sube el CSV con:
#   from google.colab import files
#   uploaded = files.upload()
# O móntalo desde Drive:
#   from google.colab import drive
#   drive.mount('/content/drive')

RUTA_CSV = "dataset_clientes_clustering.csv"

df = pd.read_csv(RUTA_CSV)
print(f"Dataset cargado: {df.shape[0]} clientes, {df.shape[1]} variables")
df.head()


# %% [2b] DETECCIÓN Y ELIMINACIÓN DE OUTLIERS
# -----------------------------------------------------------------------------
# La limpieza COMÚN del grupo (script compartido) elimina duplicados, devoluciones
# y filas sin id_cliente, pero NO elimina outliers a nivel de cliente. El PDF (pág 5)
# indica que la detección de outliers debe hacerse SOBRE EL DATASET YA CENTRADO EN
# USUARIOS, dentro de cada notebook de modelado. Aquí lo aplicamos.
#
# Sin este paso, hay clientes "ballena" (p.ej. uno con valor_total_compras > 77.000€,
# o clientes con cantidades de producto desproporcionadas) que dominan completamente
# la varianza, hacen que PCA colapse en una sola componente y generan clusters
# inútiles del tipo [4338 clientes en uno + 1 cliente solo en otro].
#
# Estrategia (siguiendo recomendación del PDF), en DOS PASOS:
#   1) Recorte por percentil 99 en variables monetarias y de cantidad: elimina
#      los extremos más radicales que distorsionan PCA aunque sean válidos.
#   2) Isolation Forest sobre el resto para outliers multivariantes más sutiles.

from sklearn.ensemble import IsolationForest

n_inicial = len(df)

# --- Paso 1: corte por percentil 99 en variables clave ---
vars_recorte = [
    'valor_total_compras', 'cantidad_total_productos', 'total_compras',
    'valor_medio_por_compra', 'valor_mensual_medio',
    'compras_mensuales_medias', 'cantidad_media_por_compra',
]
vars_recorte = [v for v in vars_recorte if v in df.columns]

print("Recorte por percentil 99 en variables clave:")
mask = pd.Series([True] * len(df), index=df.index)
for v in vars_recorte:
    p99 = df[v].quantile(0.99)
    n_descartados = (df[v] > p99).sum()
    mask &= df[v] <= p99
    print(f"  {v}: P99={p99:.2f} → descarta {n_descartados}")

df = df[mask].reset_index(drop=True)
print(f"Tras paso 1: {len(df)} clientes (eliminados {n_inicial - len(df)})")

# --- Paso 2: Isolation Forest sobre lo que queda ---
num_cols_outliers = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c != 'id_cliente']

iso_forest = IsolationForest(
    contamination=0.05,
    random_state=RANDOM_STATE,
    n_estimators=200
)
outlier_labels = iso_forest.fit_predict(df[num_cols_outliers])
n_outliers = (outlier_labels == -1).sum()
print(f"\nIsolation Forest: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")

df = df[outlier_labels == 1].reset_index(drop=True)

print(f"\nDataset final tras limpieza de outliers: {len(df)} clientes")
print(f"Total eliminado: {n_inicial - len(df)} ({(n_inicial - len(df))/n_inicial*100:.2f}%)")


# %% [3] PREPARACIÓN DE FEATURES
# -----------------------------------------------------------------------------
# Quitamos columnas no útiles para el modelo:
#  - id_cliente: identificador (lo guardamos aparte)
#  - primer_compra / ultima_compra: ya están resumidas en 'dias_como_cliente'
#    y 'meses_activo'

ids_cliente = df['id_cliente'].copy()
columnas_excluir = ['id_cliente', 'primer_compra', 'ultima_compra']
df_features = df.drop(columns=columnas_excluir).copy()

cat_cols = df_features.select_dtypes(include=['object']).columns.tolist()
num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

print(f"Variables categóricas ({len(cat_cols)}): {cat_cols}")
print(f"Variables numéricas   ({len(num_cols)})")


# %% [4] CODIFICACIÓN DE VARIABLES CATEGÓRICAS (One-Hot)
# -----------------------------------------------------------------------------
# Comprobamos cardinalidad para no explotar el nº de columnas.
for col in cat_cols:
    print(f"\n{col}: {df_features[col].nunique()} valores únicos")
    print(df_features[col].value_counts().head(5))

# Agrupamos categorías muy poco frecuentes (<1%) en "Other"
UMBRAL = 0.01
for col in cat_cols:
    freq = df_features[col].value_counts(normalize=True)
    raros = freq[freq < UMBRAL].index
    if len(raros) > 0:
        df_features[col] = df_features[col].replace(raros, 'Other')
        print(f"\n{col}: agrupadas {len(raros)} categorías → ahora {df_features[col].nunique()}")

# One-Hot
df_encoded = pd.get_dummies(df_features, columns=cat_cols, drop_first=False)
# Convertir bools de get_dummies a int
bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print(f"\nShape tras One-Hot: {df_encoded.shape}")


# %% [5] ESCALADO (RobustScaler)
# -----------------------------------------------------------------------------
# RobustScaler usa mediana e IQR → robusto a outliers residuales que pudieran
# quedar tras la limpieza. Es la opción que recomienda el PDF.

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_encoded)
X_scaled = pd.DataFrame(X_scaled, columns=df_encoded.columns)

print(f"Shape tras escalado: {X_scaled.shape}")
print("\nMuestra de variables escaladas (mediana ~0):")
print(X_scaled.describe().T[['50%', '25%', '75%']].head(8))


# %% [6] MATRIZ DE CORRELACIÓN
# -----------------------------------------------------------------------------
# Visualizamos correlaciones para justificar el uso de PCA.

plt.figure(figsize=(16, 12))
corr = X_scaled.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=False,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.7})
plt.title("Matriz de correlación", fontsize=14)
plt.tight_layout()
plt.show()

# Pares con correlación alta (|r| > 0.7) - el PDF dice que >0.5 ya es fuerte
pares_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.7:
            pares_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

print(f"\nPares con |correlación| > 0.7: {len(pares_corr)}")
for p in pares_corr[:10]:
    print(f"  {p[0]} <-> {p[1]}: {p[2]:.3f}")


# %% [7] PCA - Análisis de varianza explicada
# -----------------------------------------------------------------------------
# Buscamos el nº de componentes que cubre al menos un 80% de varianza
# (el PDF pide >60% mínimo, apuntamos más alto).

pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

var_explicada = pca_full.explained_variance_ratio_
var_acumulada = np.cumsum(var_explicada)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(var_explicada)+1), var_explicada,
            alpha=0.7, color='steelblue')
axes[0].set_xlabel("Componente principal")
axes[0].set_ylabel("Varianza explicada")
axes[0].set_title("Varianza explicada por componente")

axes[1].plot(range(1, len(var_acumulada)+1), var_acumulada,
             marker='o', color='darkorange')
axes[1].axhline(0.60, color='red', linestyle='--', label='60% (mínimo PDF)')
axes[1].axhline(0.80, color='green', linestyle='--', label='80% (objetivo)')
axes[1].set_xlabel("Nº de componentes")
axes[1].set_ylabel("Varianza acumulada")
axes[1].set_title("Varianza acumulada")
axes[1].legend()

plt.tight_layout()
plt.show()

# Nº de componentes para >=80%, con un mínimo de 3 (para poder visualizar 3D)
N_COMPONENTES = max(3, int(np.argmax(var_acumulada >= 0.80) + 1))
print(f"\nComponentes para >=80% de varianza: {N_COMPONENTES}")
print(f"Varianza explicada acumulada con {N_COMPONENTES} componentes: "
      f"{var_acumulada[N_COMPONENTES-1]*100:.2f}%")

# Aplicamos PCA
pca = PCA(n_components=N_COMPONENTES, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"\nShape tras PCA: {X_pca.shape}")


# %% [8] DENDROGRAMA
# -----------------------------------------------------------------------------
# El dendrograma es la herramienta clásica del clustering jerárquico para
# decidir el nº de clusters. Buscamos el "salto" más grande entre fusiones.
# Con ~4000 puntos truncamos para que sea legible.

print("Calculando matriz de enlace... (puede tardar 20-40s)")
Z = linkage(X_pca, method='ward')

plt.figure(figsize=(14, 6))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
)
plt.title("Dendrograma (Ward linkage) - últimas 30 fusiones", fontsize=14)
plt.xlabel("Tamaño del cluster (entre paréntesis)")
plt.ylabel("Distancia de fusión")
plt.axhline(y=Z[-3, 2], color='green', linestyle='--', alpha=0.5, label='Corte K=3')
plt.axhline(y=Z[-4, 2], color='red', linestyle='--', alpha=0.5, label='Corte K=4')
plt.axhline(y=Z[-5, 2], color='purple', linestyle='--', alpha=0.5, label='Corte K=5')
plt.legend()
plt.tight_layout()
plt.show()


# %% [9] BÚSQUEDA DEL K ÓPTIMO
# -----------------------------------------------------------------------------
# Probamos K de 2 a 10 con tres criterios:
#   - Silhouette: cuanto mayor mejor (rango -1 a 1)
#   - Davies-Bouldin: cuanto menor mejor
#   - Inercia intra-cluster: para método del codo

K_RANGE = range(2, 11)
resultados = []

for k in K_RANGE:
    modelo = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = modelo.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)

    inercia = 0
    for cid in np.unique(labels):
        puntos = X_pca[labels == cid]
        centroide = puntos.mean(axis=0)
        inercia += ((puntos - centroide) ** 2).sum()

    tamanos = pd.Series(labels).value_counts()
    resultados.append({
        'K': k,
        'silhouette': sil,
        'davies_bouldin': db,
        'inercia': inercia,
        'tam_min': tamanos.min(),
        'tam_max': tamanos.max(),
    })
    print(f"K={k}: Silhouette={sil:.4f} | DB={db:.4f} | "
          f"Tam min/max={tamanos.min()}/{tamanos.max()}")

df_resultados = pd.DataFrame(resultados)


# %% [10] VISUALIZACIÓN DE LAS MÉTRICAS
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(df_resultados['K'], df_resultados['silhouette'],
             marker='o', color='steelblue', linewidth=2)
axes[0].set_xlabel("Nº de clusters (K)")
axes[0].set_ylabel("Silhouette Score")
axes[0].set_title("Silhouette (mayor = mejor)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_resultados['K'], df_resultados['davies_bouldin'],
             marker='o', color='darkorange', linewidth=2)
axes[1].set_xlabel("Nº de clusters (K)")
axes[1].set_ylabel("Davies-Bouldin")
axes[1].set_title("Davies-Bouldin (menor = mejor)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(df_resultados['K'], df_resultados['inercia'],
             marker='o', color='seagreen', linewidth=2)
axes[2].set_xlabel("Nº de clusters (K)")
axes[2].set_ylabel("Inercia intra-cluster")
axes[2].set_title("Método del codo")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nResumen:")
print(df_resultados.to_string(index=False))


# %% [11] COMPARACIÓN DE LINKAGES
# -----------------------------------------------------------------------------
# AgglomerativeClustering acepta varios criterios de fusión:
#   - ward: minimiza varianza intra-cluster (solo euclídea)
#   - complete: máxima distancia entre puntos
#   - average: distancia media
#   - single: distancia mínima (suele formar "cadenas" alargadas)
#
# Elegimos K basándonos en lo que hayamos visto en [9] y [10].
# Cámbialo según lo que mejor se vea:

K_OPTIMO = int(df_resultados.loc[df_resultados['silhouette'].idxmax(), 'K'])
print(f"K elegido por silueta máxima: {K_OPTIMO}")
# Si visualmente prefieres otro, sobrescribe:
# K_OPTIMO = 4

linkages_lista = ['ward', 'complete', 'average', 'single']
comparacion = []

for link in linkages_lista:
    modelo = AgglomerativeClustering(n_clusters=K_OPTIMO, linkage=link)
    labels = modelo.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    tamanos = pd.Series(labels).value_counts().sort_index().tolist()
    comparacion.append({
        'linkage': link,
        'silhouette': sil,
        'davies_bouldin': db,
        'tamanos': tamanos,
    })

df_comparacion = pd.DataFrame(comparacion)
print(f"\nComparativa de linkages (K={K_OPTIMO}):")
print(df_comparacion.to_string(index=False))


# %% [12] MODELO FINAL
# -----------------------------------------------------------------------------
# Elegir el linkage que mejor combine silueta alta + tamaños equilibrados
# (el PDF avisa: un cluster con el 95% de los datos no aporta valor).

LINKAGE_FINAL = 'ward'   # cámbialo si la comparativa sugiere otro
print(f"\nModelo final: AgglomerativeClustering(n_clusters={K_OPTIMO}, "
      f"linkage='{LINKAGE_FINAL}')")

modelo_final = AgglomerativeClustering(n_clusters=K_OPTIMO, linkage=LINKAGE_FINAL)
labels_final = modelo_final.fit_predict(X_pca)

sil_final = silhouette_score(X_pca, labels_final)
db_final = davies_bouldin_score(X_pca, labels_final)

print(f"Silhouette Score final: {sil_final:.4f}")
print(f"Davies-Bouldin final:   {db_final:.4f}")
print(f"\nDistribución de clusters:")
print(pd.Series(labels_final).value_counts().sort_index())


# %% [13] VISUALIZACIÓN 2D (PCA - 2 primeras componentes)
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
palette = sns.color_palette("Set2", K_OPTIMO)

for i, cluster in enumerate(sorted(np.unique(labels_final))):
    mask = labels_final == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=[palette[i]], label=f"Cluster {cluster} (n={mask.sum()})",
                alpha=0.6, edgecolors='w', s=50)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
plt.title(f"Clusters en espacio PCA 2D (Ward, K={K_OPTIMO})", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


# %% [14] VISUALIZACIÓN 3D
# -----------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for i, cluster in enumerate(sorted(np.unique(labels_final))):
    mask = labels_final == cluster
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
               c=[palette[i]], label=f"Cluster {cluster}", alpha=0.6, s=40)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title(f"Clusters en PCA 3D (K={K_OPTIMO})", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()


# %% [15] ANÁLISIS DE SILUETA POR CLUSTER
# -----------------------------------------------------------------------------
sil_samples = silhouette_samples(X_pca, labels_final)

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
for i in range(K_OPTIMO):
    cluster_sil = sil_samples[labels_final == i]
    cluster_sil.sort()
    size = cluster_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     facecolor=palette[i], edgecolor=palette[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, f'Cluster {i}')
    y_lower = y_upper + 10

ax.axvline(x=sil_final, color="red", linestyle="--",
           label=f"Media: {sil_final:.3f}")
ax.set_xlabel("Coeficiente de silueta")
ax.set_ylabel("Cluster")
ax.set_title("Análisis de silueta por cluster", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()


# %% [16] INTERPRETACIÓN - Perfil medio por cluster
# -----------------------------------------------------------------------------
df_interpret = df_features.copy()
df_interpret['cluster'] = labels_final

vars_clave = [
    'total_compras', 'valor_total_compras', 'valor_medio_por_compra',
    'cantidad_total_productos', 'dias_como_cliente',
    'frecuencia_compra', 'meses_activo', 'valor_mensual_medio',
    'porcentaje_compras_fines_semana', 'diversidad_categorias',
    'total_productos_diferentes',
]
vars_clave = [v for v in vars_clave if v in df_interpret.columns]

perfil = df_interpret.groupby('cluster')[vars_clave].mean().round(2)
perfil['n_clientes'] = df_interpret.groupby('cluster').size()
print("Perfil medio por cluster:")
print(perfil.T)


# %% [17] HEATMAP DEL PERFIL
# -----------------------------------------------------------------------------
perfil_norm = perfil.drop(columns='n_clientes')
perfil_z = (perfil_norm - perfil_norm.mean()) / perfil_norm.std()

plt.figure(figsize=(12, max(4, K_OPTIMO * 0.8)))
sns.heatmap(perfil_z, cmap='RdBu_r', center=0, annot=perfil_norm.round(1),
            fmt='.1f', linewidths=0.5, cbar_kws={'label': 'Z-score'})
plt.title("Perfil de clusters (color = z-score, anotación = valor real)",
          fontsize=13)
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()


# %% [18] TOP CARACTERÍSTICAS DIFERENCIADORAS POR CLUSTER
# -----------------------------------------------------------------------------
print("=" * 70)
print("TOP 5 CARACTERÍSTICAS DIFERENCIADORAS POR CLUSTER")
print("=" * 70)
for c in range(K_OPTIMO):
    print(f"\n--- Cluster {c} (n={int(perfil.loc[c, 'n_clientes'])}) ---")
    diferencias = perfil_z.loc[c].sort_values(key=abs, ascending=False).head(5)
    for var, z in diferencias.items():
        valor_real = perfil_norm.loc[c, var]
        flecha = "↑ ALTO" if z > 0 else "↓ BAJO"
        print(f"  {flecha:7} {var}: {valor_real:.2f}  (z={z:+.2f})")


# %% [19] EXPORTAR RESULTADOS
# -----------------------------------------------------------------------------
df_export = pd.DataFrame({
    'id_cliente': ids_cliente,
    'cluster': labels_final,
})
df_export.to_csv("clientes_con_cluster_jerarquico.csv", index=False)
print(f"\nExportado: clientes_con_cluster_jerarquico.csv ({len(df_export)} filas)")

perfil.to_csv("perfil_clusters_jerarquico.csv")
print("Exportado: perfil_clusters_jerarquico.csv")


# %% [20] RESUMEN FINAL
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN DEL MODELO FINAL")
print("=" * 70)
print(f"Algoritmo:         AgglomerativeClustering")
print(f"Linkage:           {LINKAGE_FINAL}")
print(f"Nº clusters (K):   {K_OPTIMO}")
print(f"Nº clientes:       {len(labels_final)}")
print(f"Silhouette Score:  {sil_final:.4f}")
print(f"Davies-Bouldin:    {db_final:.4f}")
print(f"Componentes PCA:   {N_COMPONENTES} ({var_acumulada[N_COMPONENTES-1]*100:.1f}% varianza)")
print("=" * 70)
