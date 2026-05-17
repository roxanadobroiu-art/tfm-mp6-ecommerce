"""
Clustering de Clientes con HDBSCAN para Estrategias de Marketing y Ventas
Objetivo: Segmentar clientes para personalizar ofertas, promociones y comunicación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import hdbscan
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def cargar_y_explorar_datos(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    
    print(f"📊 Dataset: {df.shape[0]} clientes × {df.shape[1]} características")
    
    columnas_clave = ['total_compras', 'valor_total_compras', 'valor_medio_por_compra', 
                      'frecuencia_compra', 'dias_como_cliente']
    print(f"\n{df[columnas_clave].describe().round(2)}")
    
    return df

def preprocesar_para_clustering(df):
    variables_marketing = [
        'total_compras',
        'valor_total_compras',
        'valor_medio_por_compra',
        'frecuencia_compra',
        'dias_como_cliente',
        'cantidad_total_productos',
        'porcentaje_compras_fines_semana',
        'meses_activo',
        'valor_mensual_medio'
    ]
    
    variables_disponibles = [var for var in variables_marketing if var in df.columns]
    df_clustering = df[variables_disponibles].copy()
    
    if df_clustering.isnull().sum().sum() > 0:
        for col in df_clustering.columns:
            if df_clustering[col].isnull().sum() > 0:
                df_clustering[col].fillna(df_clustering[col].median(), inplace=True)
    
    print(f"Variables para clustering: {df_clustering.shape[1]}")
    
    return df_clustering

def aplicar_escalado(df_clustering):
    scaler = StandardScaler()
    df_escalado = scaler.fit_transform(df_clustering)
    return df_escalado, scaler

def encontrar_parametros_optimos(datos_escalados):
    print("\n🔍 Buscando parámetros óptimos...")
    
    min_cluster_sizes = [10, 15, 20, 25, 30, 40, 50]
    min_samples_values = [1, 3, 5, 10]
    
    resultados = []
    total_clientes = len(datos_escalados)
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            etiquetas = clusterer.fit_predict(datos_escalados)
            
            n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
            n_ruido = list(etiquetas).count(-1)
            porcentaje_ruido = (n_ruido / total_clientes) * 100
            
            if n_clusters > 1:
                tamaños_clusters = []
                for cluster_id in set(etiquetas):
                    if cluster_id != -1:
                        tamaños_clusters.append(list(etiquetas).count(cluster_id))
                
                equilibrio = np.std(tamaños_clusters) / np.mean(tamaños_clusters)
                
                if n_ruido < total_clientes:
                    indices_validos = np.array(etiquetas) != -1
                    if np.sum(indices_validos) > 1 and n_clusters > 1:
                        silhouette = silhouette_score(
                            datos_escalados[indices_validos], 
                            np.array(etiquetas)[indices_validos]
                        )
                        davies_bouldin = davies_bouldin_score(
                            datos_escalados[indices_validos], 
                            np.array(etiquetas)[indices_validos]
                        )
                        calinski_harabasz = calinski_harabasz_score(
                            datos_escalados[indices_validos], 
                            np.array(etiquetas)[indices_validos]
                        )
                    else:
                        silhouette = davies_bouldin = calinski_harabasz = -1
                else:
                    silhouette = davies_bouldin = calinski_harabasz = -1
            else:
                equilibrio = float('inf')
                silhouette = davies_bouldin = calinski_harabasz = -1
            
            resultados.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_ruido': n_ruido,
                'porcentaje_ruido': porcentaje_ruido,
                'equilibrio_clusters': equilibrio,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz
            })
    
    df_resultados = pd.DataFrame(resultados)
    
    configs_validas = df_resultados[
        (df_resultados['n_clusters'] >= 2) & 
        (df_resultados['n_clusters'] <= 8) &
        (df_resultados['porcentaje_ruido'] <= 30)
    ].copy()
    
    if len(configs_validas) > 0:
        configs_validas['puntuacion'] = (
            configs_validas['silhouette_score'] * 0.3 +
            (1 / (configs_validas['davies_bouldin_score'] + 0.1)) * 0.2 +
            (configs_validas['calinski_harabasz_score'] / configs_validas['calinski_harabasz_score'].max()) * 0.2 +
            (1 / (configs_validas['equilibrio_clusters'] + 0.1)) * 0.2 +
            ((100 - configs_validas['porcentaje_ruido']) / 100) * 0.1
        )
        
        top_configs = configs_validas.nlargest(5, 'puntuacion')
        
        print("\n🏆 TOP 5 CONFIGURACIONES:")
        print(top_configs[['min_cluster_size', 'min_samples', 'n_clusters', 'porcentaje_ruido', 
                           'silhouette_score', 'davies_bouldin_score']].round(3))
        
        mejor_config = top_configs.iloc[0]
        print(f"\n✨ CONFIGURACIÓN ÓPTIMA:")
        print(f"   Min cluster size: {int(mejor_config['min_cluster_size'])} | Min samples: {int(mejor_config['min_samples'])}")
        print(f"   Clusters: {int(mejor_config['n_clusters'])} | Ruido: {mejor_config['porcentaje_ruido']:.1f}%")
        print(f"   Silueta: {mejor_config['silhouette_score']:.3f} | Davies-Bouldin: {mejor_config['davies_bouldin_score']:.3f}")
        
        return int(mejor_config['min_cluster_size']), int(mejor_config['min_samples'])
    else:
        print("⚠️ No se encontraron configuraciones óptimas. Usando valores por defecto.")
        return 20, 5

def aplicar_clustering_final(datos_escalados, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    etiquetas_clusters = clusterer.fit_predict(datos_escalados)
    
    n_clusters = len(set(etiquetas_clusters)) - (1 if -1 in etiquetas_clusters else 0)
    n_ruido = list(etiquetas_clusters).count(-1)
    total_clientes = len(etiquetas_clusters)
    
    print(f"\n✅ Clustering completado: {n_clusters} segmentos | {n_ruido} atípicos ({n_ruido/total_clientes*100:.1f}%)")
    
    if n_clusters >= 2:
        indices_validos = etiquetas_clusters != -1
        if np.sum(indices_validos) > 1:
            silhouette = silhouette_score(datos_escalados[indices_validos], etiquetas_clusters[indices_validos])
            davies_bouldin = davies_bouldin_score(datos_escalados[indices_validos], etiquetas_clusters[indices_validos])
            calinski_harabasz = calinski_harabasz_score(datos_escalados[indices_validos], etiquetas_clusters[indices_validos])
            
            print(f"\n📊 MÉTRICAS:")
            print(f"   Silueta: {silhouette:.4f} | Davies-Bouldin: {davies_bouldin:.4f} | Calinski-Harabasz: {calinski_harabasz:.2f}")
            
            calidad_sil = "Excelente" if silhouette >= 0.7 else "Buena" if silhouette >= 0.5 else "Moderada" if silhouette >= 0.25 else "Débil"
            calidad_db = "Excelente" if davies_bouldin <= 0.5 else "Buena" if davies_bouldin <= 1.0 else "Moderada" if davies_bouldin <= 2.0 else "Débil"
            print(f"   Separación: {calidad_sil} | Compacidad: {calidad_db}")

    print(f"\n📈 DISTRIBUCIÓN:")
    for cluster_id in sorted(set(etiquetas_clusters)):
        count = list(etiquetas_clusters).count(cluster_id)
        percentage = (count / total_clientes) * 100
        label = "Ruido" if cluster_id == -1 else f"Segmento {cluster_id}"
        print(f"   {label}: {count} clientes ({percentage:.1f}%)")
    
    return etiquetas_clusters, clusterer

def analizar_segmentos_marketing(df_original, df_clustering, etiquetas_clusters):
    print(f"\n📊 ANÁLISIS DE SEGMENTOS")
    print("=" * 50)
    
    df_con_clusters = df_original.copy()
    df_con_clusters['segmento'] = etiquetas_clusters
    
    segmentos_validos = [seg for seg in set(etiquetas_clusters) if seg != -1]
    
    for segmento in sorted(segmentos_validos):
        datos_segmento = df_con_clusters[df_con_clusters['segmento'] == segmento]
        n_clientes = len(datos_segmento)
        porcentaje = (n_clientes / len(df_con_clusters)) * 100
        
        print(f"\n🎯 SEGMENTO {segmento} ({n_clientes} clientes - {porcentaje:.1f}%)")
        print("-" * 40)
        
        print(f"   Compras: {datos_segmento['total_compras'].mean():.2f}")
        print(f"   Valor Total: €{datos_segmento['valor_total_compras'].mean():.2f}")
        print(f"   Ticket Promedio: €{datos_segmento['valor_medio_por_compra'].mean():.2f}")
        print(f"   Frecuencia: {datos_segmento['frecuencia_compra'].mean():.4f}")
        print(f"   Antigüedad: {datos_segmento['dias_como_cliente'].mean():.0f} días")
        
        perfil = clasificar_segmento_marketing(datos_segmento, df_con_clusters)
        estrategia = generar_estrategia_marketing(perfil, {})
        print(f"   Perfil: {perfil}")
        print(f"   Estrategia: {estrategia}")
    
    return df_con_clusters

def clasificar_segmento_marketing(datos_segmento, df_completo):
    valor_p75 = df_completo['valor_total_compras'].quantile(0.75)
    freq_p75 = df_completo['frecuencia_compra'].quantile(0.75)
    valor_p25 = df_completo['valor_total_compras'].quantile(0.25)
    freq_p25 = df_completo['frecuencia_compra'].quantile(0.25)
    
    valor_medio = datos_segmento['valor_total_compras'].mean()
    freq_media = datos_segmento['frecuencia_compra'].mean()
    antiguedad_media = datos_segmento['dias_como_cliente'].mean()
    
    if valor_medio >= valor_p75 and freq_media >= freq_p75:
        return "🌟 VIP - Alto Valor y Alta Frecuencia"
    elif valor_medio >= valor_p75 and freq_media < freq_p75:
        return "💎 Premium - Alto Valor, Baja Frecuencia"
    elif valor_medio < valor_p75 and freq_media >= freq_p75:
        return "🔄 Frecuentes - Bajo Valor, Alta Frecuencia"
    elif valor_medio <= valor_p25 and antiguedad_media <= 90:
        return "🆕 Nuevos - Recién Llegados"
    elif valor_medio <= valor_p25:
        return "😴 Inactivos - Bajo Engagement"
    else:
        return "👥 Regulares - Comportamiento Estándar"

def generar_estrategia_marketing(perfil, metricas):
    estrategias = {
        "🌟 VIP - Alto Valor y Alta Frecuencia": "Programas de lealtad premium, acceso anticipado a productos",
        "💎 Premium - Alto Valor, Baja Frecuencia": "Ofertas exclusivas, comunicación de alta calidad, eventos VIP",
        "🔄 Frecuentes - Bajo Valor, Alta Frecuencia": "Cross-selling, up-selling, descuentos por volumen",
        "🆕 Nuevos - Recién Llegados": "Ofertas de bienvenida, onboarding personalizado",
        "😴 Inactivos - Bajo Engagement": "Campañas de reactivación, descuentos agresivos",
        "👥 Regulares - Comportamiento Estándar": "Marketing mix equilibrado, promociones estacionales"
    }
    
    return estrategias.get(perfil, "Estrategia personalizada")

def evaluar_calidad_clustering(datos_escalados, etiquetas_clusters, df_clustering):
    print(f"\n🔬 EVALUACIÓN DE CALIDAD")
    print("=" * 60)
    
    indices_validos = np.array(etiquetas_clusters) != -1
    datos_limpios = datos_escalados[indices_validos]
    etiquetas_limpias = np.array(etiquetas_clusters)[indices_validos]
    n_clusters = len(set(etiquetas_limpias))
    
    if n_clusters < 2:
        print("⚠️ No hay suficientes clusters para evaluar calidad")
        return
    
    silhouette_global = silhouette_score(datos_limpios, etiquetas_limpias)
    davies_bouldin = davies_bouldin_score(datos_limpios, etiquetas_limpias)
    calinski_harabasz = calinski_harabasz_score(datos_limpios, etiquetas_limpias)
    
    print(f"📊 MÉTRICAS GLOBALES:")
    print(f"   Silueta: {silhouette_global:.4f} | Davies-Bouldin: {davies_bouldin:.4f} | Calinski-Harabasz: {calinski_harabasz:.2f}")
    
    silhouette_samples_scores = silhouette_samples(datos_limpios, etiquetas_limpias)
    
    print(f"\n🔍 ANÁLISIS POR CLUSTER:")
    for cluster_id in sorted(set(etiquetas_limpias)):
        cluster_silhouette = silhouette_samples_scores[etiquetas_limpias == cluster_id]
        print(f"   Cluster {cluster_id}: {len(cluster_silhouette)} puntos, Silueta = {cluster_silhouette.mean():.3f}")
    
    crear_visualizaciones_evaluacion(datos_escalados, etiquetas_clusters, df_clustering, 
                                    silhouette_samples_scores, etiquetas_limpias)

def crear_visualizaciones_evaluacion(datos_escalados, etiquetas_clusters, df_clustering, 
                                    silhouette_samples_scores, etiquetas_limpias):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Análisis de Silhouette
    y_lower = 10
    colores = plt.cm.nipy_spectral(np.linspace(0, 1, len(set(etiquetas_limpias))))
    
    for cluster_id, color in zip(sorted(set(etiquetas_limpias)), colores):
        cluster_scores = silhouette_samples_scores[etiquetas_limpias == cluster_id]
        cluster_scores.sort()
        
        size = cluster_scores.shape[0]
        y_upper = y_lower + size
        
        axes[0, 0].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                                 facecolor=color, edgecolor=color, alpha=0.7)
        axes[0, 0].text(-0.05, y_lower + 0.5 * size, str(cluster_id))
        y_lower = y_upper + 10
    
    axes[0, 0].set_xlabel('Valores de Silhouette')
    axes[0, 0].set_ylabel('Cluster')
    axes[0, 0].set_title('Análisis de Silhouette')
    
    silhouette_avg = silhouette_samples_scores.mean()
    axes[0, 0].axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Promedio: {silhouette_avg:.3f}')
    axes[0, 0].legend()
    
    # 2. PCA 2D
    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(datos_escalados)
    
    scatter = axes[0, 1].scatter(datos_pca[:, 0], datos_pca[:, 1], 
                                c=etiquetas_clusters, cmap='tab10', alpha=0.7, s=50)
    
    indices_validos = np.array(etiquetas_clusters) != -1
    for cluster_id in sorted(set(etiquetas_clusters)):
        if cluster_id != -1:
            cluster_points = datos_pca[np.array(etiquetas_clusters) == cluster_id]
            centroid = cluster_points.mean(axis=0)
            axes[0, 1].scatter(centroid[0], centroid[1], c='red', marker='x', s=200, linewidths=3)
            
            distances = np.sqrt(((cluster_points - centroid) ** 2).sum(axis=1))
            radius = distances.std()
            circle = plt.Circle(centroid, radius, fill=False, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].add_patch(circle)
    
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 1].set_title('Separación de Clusters (PCA)')
    
    # 3. Distancias intra vs inter-cluster
    distancias_intra = []
    distancias_inter = []
    
    for cluster_id in sorted(set(etiquetas_clusters)):
        if cluster_id != -1:
            cluster_points = datos_escalados[np.array(etiquetas_clusters) == cluster_id]
            if len(cluster_points) > 1:
                intra_dist = pdist(cluster_points)
                distancias_intra.extend(intra_dist)
                
                for other_cluster in sorted(set(etiquetas_clusters)):
                    if other_cluster != -1 and other_cluster != cluster_id:
                        other_points = datos_escalados[np.array(etiquetas_clusters) == other_cluster]
                        for point in cluster_points[:min(50, len(cluster_points))]:
                            for other_point in other_points[:min(50, len(other_points))]:
                                inter_dist = np.linalg.norm(point - other_point)
                                distancias_inter.append(inter_dist)
    
    if distancias_intra and distancias_inter:
        axes[0, 2].hist(distancias_intra, bins=30, alpha=0.7, label='Intra-cluster', color='blue')
        axes[0, 2].hist(distancias_inter, bins=30, alpha=0.7, label='Inter-cluster', color='red')
        axes[0, 2].set_xlabel('Distancia')
        axes[0, 2].set_ylabel('Frecuencia')
        axes[0, 2].set_title('Distribución de Distancias')
        axes[0, 2].legend()
        
        mean_intra = np.mean(distancias_intra)
        mean_inter = np.mean(distancias_inter)
        separacion_ratio = mean_inter / mean_intra if mean_intra > 0 else 0
        
        axes[0, 2].axvline(mean_intra, color='blue', linestyle='--')
        axes[0, 2].axvline(mean_inter, color='red', linestyle='--')
        axes[0, 2].text(0.05, 0.95, f'Ratio: {separacion_ratio:.2f}', 
                       transform=axes[0, 2].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Matriz de distancias entre centroides
    centroides = []
    cluster_ids_validos = []
    
    for cluster_id in sorted(set(etiquetas_clusters)):
        if cluster_id != -1:
            cluster_data = datos_escalados[np.array(etiquetas_clusters) == cluster_id]
            centroide = cluster_data.mean(axis=0)
            centroides.append(centroide)
            cluster_ids_validos.append(cluster_id)
    
    if len(centroides) > 1:
        distancias = squareform(pdist(centroides))
        
        im = axes[1, 0].imshow(distancias, cmap='viridis', interpolation='nearest')
        axes[1, 0].set_xticks(range(len(cluster_ids_validos)))
        axes[1, 0].set_yticks(range(len(cluster_ids_validos)))
        axes[1, 0].set_xticklabels(cluster_ids_validos)
        axes[1, 0].set_yticklabels(cluster_ids_validos)
        axes[1, 0].set_title('Distancias entre Centroides')
        plt.colorbar(im, ax=axes[1, 0])
        
        for i in range(len(cluster_ids_validos)):
            for j in range(len(cluster_ids_validos)):
                axes[1, 0].text(j, i, f'{distancias[i, j]:.2f}', 
                               ha='center', va='center', 
                               color='white' if distancias[i, j] > distancias.mean() else 'black')
    
    # 5. Boxplot de calidad
    silhouette_por_cluster = []
    clusters_para_box = []
    
    for cluster_id in sorted(set(etiquetas_limpias)):
        scores = silhouette_samples_scores[etiquetas_limpias == cluster_id]
        silhouette_por_cluster.extend(scores)
        clusters_para_box.extend([cluster_id] * len(scores))
    
    if silhouette_por_cluster:
        df_sil = pd.DataFrame({'silhouette': silhouette_por_cluster, 'cluster': clusters_para_box})
        unique_clusters = sorted(set(clusters_para_box))
        box_data = [df_sil[df_sil['cluster'] == c]['silhouette'].values for c in unique_clusters]
        
        axes[1, 1].boxplot(box_data, labels=unique_clusters)
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Silhouette Score')
        axes[1, 1].set_title('Calidad por Cluster')
        axes[1, 1].axhline(y=silhouette_avg, color='red', linestyle='--', alpha=0.7)
    
    # 6. Resumen
    axes[1, 2].axis('off')
    
    silhouette_global = silhouette_samples_scores.mean()
    davies_bouldin = davies_bouldin_score(datos_escalados[indices_validos], etiquetas_limpias)
    n_clusters_actual = len(set(etiquetas_limpias))
    
    resumen_texto = f"""
RESUMEN

📊 Clusters: {n_clusters_actual}
🎯 Silueta: {silhouette_global:.3f}
📉 Davies-Bouldin: {davies_bouldin:.3f}
    """
    
    axes[1, 2].text(0.05, 0.95, resumen_texto, transform=axes[1, 2].transAxes, 
                   verticalalignment='top', fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('EVALUACIÓN DE CALIDAD DEL CLUSTERING', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def crear_visualizaciones_negocio(df_clustering, etiquetas_clusters):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribución de segmentos
    valores_clusters = pd.Series(etiquetas_clusters).value_counts().sort_index()
    colores = plt.cm.Set3(np.linspace(0, 1, len(valores_clusters)))
    
    bars = axes[0, 0].bar(valores_clusters.index, valores_clusters.values, color=colores)
    axes[0, 0].set_title('Distribución por Segmento', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Segmento')
    axes[0, 0].set_ylabel('Clientes')
    
    for bar, valor in zip(bars, valores_clusters.values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(valores_clusters.values) * 0.01,
                       f'{valor}\n({valor/sum(valores_clusters.values)*100:.1f}%)',
                       ha='center', va='bottom')
    
    # 2. Valor vs Frecuencia
    scatter = axes[0, 1].scatter(df_clustering['valor_total_compras'], 
                                df_clustering['frecuencia_compra'],
                                c=etiquetas_clusters, cmap='tab10', alpha=0.7)
    axes[0, 1].set_title('Valor Total vs Frecuencia', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Valor Total (€)')
    axes[0, 1].set_ylabel('Frecuencia')
    plt.colorbar(scatter, ax=axes[0, 1], label='Segmento')
    
    # 3. Ticket promedio por segmento
    df_temp = df_clustering.copy()
    df_temp['segmento'] = etiquetas_clusters
    df_temp_clean = df_temp[df_temp['segmento'] != -1]
    
    if not df_temp_clean.empty:
        df_temp_clean.boxplot(column='valor_medio_por_compra', by='segmento', ax=axes[1, 0])
        axes[1, 0].set_title('Ticket Promedio por Segmento')
        axes[1, 0].set_xlabel('Segmento')
        axes[1, 0].set_ylabel('Valor Medio (€)')
    
    # 4. Compras vs Antigüedad
    scatter2 = axes[1, 1].scatter(df_clustering['total_compras'], 
                                 df_clustering['dias_como_cliente'],
                                 c=etiquetas_clusters, cmap='tab10', alpha=0.7)
    axes[1, 1].set_title('Compras vs Antigüedad', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Total Compras')
    axes[1, 1].set_ylabel('Días como Cliente')
    plt.colorbar(scatter2, ax=axes[1, 1], label='Segmento')
    
    plt.tight_layout()
    plt.show()

def main():
    print("🚀 SEGMENTACIÓN DE CLIENTES PARA MARKETING")
    print("=" * 60)
    
    ruta_dataset = "DataSet/dataset_clientes_clustering.csv"
    
    try:
        df = cargar_y_explorar_datos(ruta_dataset)
        df_clustering = preprocesar_para_clustering(df)
        datos_escalados, scaler = aplicar_escalado(df_clustering)
        mejor_min_cluster, mejor_min_samples = encontrar_parametros_optimos(datos_escalados)
        etiquetas, clusterer = aplicar_clustering_final(datos_escalados, mejor_min_cluster, mejor_min_samples)
        evaluar_calidad_clustering(datos_escalados, etiquetas, df_clustering)
        df_final = analizar_segmentos_marketing(df, df_clustering, etiquetas)
        crear_visualizaciones_negocio(df_clustering, etiquetas)
        
        ruta_salida = "DataSet/clientes_segmentados_marketing.csv"
        df_final.to_csv(ruta_salida, index=False)
        print(f"\n💾 Resultados guardados en: {ruta_salida}")
        print("\n✅ ANÁLISIS COMPLETADO")
        
        return df_final, etiquetas
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {ruta_dataset}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    resultado = main()