import pandas as pd

RUTA_ENTRADA = "./DataSet/data.csv"

def formato_es(numero):
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


print("=== 1. CARGA DEL DATASET ORIGINAL ===")
df = pd.read_csv(RUTA_ENTRADA, encoding="latin-1")
print("Filas y columnas iniciales:", df.shape)


print("\n=== 2. CAMBIO DE NOMBRES A ESPANOL ===")
df = df.rename(
    columns={
        "InvoiceNo": "numero_factura",
        "StockCode": "codigo_producto",
        "Description": "descripcion_original",
        "Quantity": "cantidad",
        "InvoiceDate": "fecha_factura_original",
        "UnitPrice": "precio_unitario",
        "CustomerID": "id_cliente",
        "Country": "pais",
    }
)

print("Columnas actuales:")
print(df.columns.tolist())


print("\n=== 3. LIMPIEZA BASICA ===")
filas_antes = len(df)
df = df.drop_duplicates().copy()
print("Duplicados eliminados:", filas_antes - len(df))

filas_antes = len(df)
df = df[~df["numero_factura"].astype(str).str.startswith("C")].copy()
df = df[df["cantidad"] > 0].copy()
print("Devoluciones/cancelaciones eliminadas:", filas_antes - len(df))

filas_antes = len(df)
df = df.dropna(subset=["id_cliente"]).copy()
print("Filas sin id_cliente eliminadas:", filas_antes - len(df))


print("\n=== 4. PROCESAMIENTO DE FECHAS ===")
# Convertir fecha a datetime
df["fecha"] = pd.to_datetime(df["fecha_factura_original"], format="%m/%d/%Y %H:%M")
df["dia_semana"] = df["fecha"].dt.dayofweek + 1  # 1=Lunes, 7=Domingo
df["mes"] = df["fecha"].dt.month
df["anio"] = df["fecha"].dt.year
df["hora"] = df["fecha"].dt.hour
df["fin_de_semana"] = df["dia_semana"].isin([6, 7])

print("Fechas procesadas correctamente")


print("\n=== 5. VARIABLES DE NEGOCIO ===")
# Calcular importe por línea
df["importe_linea"] = df["cantidad"] * df["precio_unitario"]

# Categorización básica de productos (basada en descripción)
def categorizar_producto(descripcion):
    if pd.isna(descripcion):
        return "sin_descripcion"
    desc_lower = str(descripcion).lower()
    if any(word in desc_lower for word in ["christmas", "xmas", "santa"]):
        return "navidad"
    elif any(word in desc_lower for word in ["heart", "love", "valentine"]):
        return "romantico"
    elif any(word in desc_lower for word in ["baby", "bab", "child"]):
        return "bebe"
    elif any(word in desc_lower for word in ["birthday", "cake", "party"]):
        return "cumpleanos"
    elif any(word in desc_lower for word in ["postage", "carriage", "manual"]):
        return "no_producto"
    else:
        return "otros"

df["categoria_producto"] = df["descripcion_original"].apply(categorizar_producto)
print("Categorización de productos completada")


print("\n=== 6. GENERACIÓN DE VARIABLES CENTRADAS EN CLIENTES ===")
print("Creando dataset centrado en usuarios...")

# Crear DataFrame centrado en clientes
clientes_vars = []

for cliente_id in df["id_cliente"].unique():
    cliente_data = df[df["id_cliente"] == cliente_id].copy()
    
    # Variables básicas del cliente
    variables_cliente = {
        "id_cliente": cliente_id,
        
        # 1. VARIABLES DE VOLUMEN DE COMPRAS
        "total_compras": len(cliente_data),
        "total_productos_diferentes": cliente_data["codigo_producto"].nunique(),
        "total_facturas": cliente_data["numero_factura"].nunique(),
        
        # 2. VARIABLES DE VALOR MONETARIO  
        "valor_total_compras": cliente_data["importe_linea"].sum(),
        "valor_medio_por_compra": cliente_data["importe_linea"].mean(),
        "valor_mediano_por_compra": cliente_data["importe_linea"].median(),
        "precio_unitario_medio": cliente_data["precio_unitario"].mean(),
        
        # 3. VARIABLES DE CANTIDAD
        "cantidad_total_productos": cliente_data["cantidad"].sum(),
        "cantidad_media_por_compra": cliente_data["cantidad"].mean(),
        
        # 4. VARIABLES TEMPORALES
        "primer_compra": cliente_data["fecha"].min(),
        "ultima_compra": cliente_data["fecha"].max(),
        "dias_como_cliente": (cliente_data["fecha"].max() - cliente_data["fecha"].min()).days + 1,
        
        # 5. VARIABLES DE FRECUENCIA
        "dia_semana_favorito": cliente_data["dia_semana"].mode().iloc[0] if not cliente_data["dia_semana"].mode().empty else 1,
        "hora_favorita": cliente_data["hora"].mode().iloc[0] if not cliente_data["hora"].mode().empty else 12,
        
        # 6. VARIABLES GEOGRÁFICAS
        "pais_principal": cliente_data["pais"].mode().iloc[0] if not cliente_data["pais"].mode().empty else "Unknown",
        "paises_diferentes": cliente_data["pais"].nunique(),
        
        # 7. VARIABLES DE COMPORTAMIENTO
        "categorias_productos_diferentes": cliente_data["categoria_producto"].nunique(),
        "compras_fines_semana": len(cliente_data[cliente_data["fin_de_semana"]]),
        "porcentaje_compras_fines_semana": len(cliente_data[cliente_data["fin_de_semana"]]) / len(cliente_data) * 100,
        
        # 8. VARIABLES DE ESTACIONALIDAD
        "mes_mas_activo": cliente_data["mes"].mode().iloc[0] if not cliente_data["mes"].mode().empty else 1,
        "meses_diferentes_compra": cliente_data["mes"].nunique(),
    }
    
    # 9. VARIABLES DE FRECUENCIA TEMPORAL (si tiene más de una compra)
    if len(cliente_data) > 1:
        fechas_ordenadas = cliente_data["fecha"].sort_values()
        diferencias_dias = fechas_ordenadas.diff().dt.days.dropna()
        
        variables_cliente["tiempo_medio_entre_compras"] = diferencias_dias.mean()
        variables_cliente["tiempo_mediano_entre_compras"] = diferencias_dias.median() 
        variables_cliente["frecuencia_compra"] = len(cliente_data) / variables_cliente["dias_como_cliente"] if variables_cliente["dias_como_cliente"] > 0 else 0
    else:
        variables_cliente["tiempo_medio_entre_compras"] = 0
        variables_cliente["tiempo_mediano_entre_compras"] = 0
        variables_cliente["frecuencia_compra"] = 0
    
    # 10. VARIABLES MENSUALES (si es cliente de varios meses)
    cliente_data["anio_mes"] = cliente_data["fecha"].dt.to_period("M")
    compras_por_mes = cliente_data.groupby("anio_mes")["importe_linea"].agg(["sum", "count"])
    
    variables_cliente["meses_activo"] = len(compras_por_mes)
    variables_cliente["valor_mensual_medio"] = compras_por_mes["sum"].mean()
    variables_cliente["compras_mensuales_medias"] = compras_por_mes["count"].mean()
    
    # 11. VARIABLES DE DIVERSIDAD DE PRODUCTOS
    productos_por_categoria = cliente_data["categoria_producto"].value_counts()
    variables_cliente["categoria_favorita"] = productos_por_categoria.index[0] if len(productos_por_categoria) > 0 else "otros"
    variables_cliente["diversidad_categorias"] = len(productos_por_categoria)
    
    clientes_vars.append(variables_cliente)

# Crear DataFrame final centrado en clientes
df_clientes = pd.DataFrame(clientes_vars)

print(f"Dataset centrado en clientes creado: {len(df_clientes)} clientes únicos")
print(f"Variables generadas: {len(df_clientes.columns)} columnas")


print("\n=== 7. RESUMEN DE VARIABLES GENERADAS ===")
print("\nVariables de Volumen:")
print("- total_compras, total_productos_diferentes, total_facturas")

print("\nVariables de Valor Monetario:")
print("- valor_total_compras, valor_medio_por_compra, valor_mediano_por_compra")

print("\nVariables Temporales:")
print("- dias_como_cliente, tiempo_medio_entre_compras, frecuencia_compra")

print("\nVariables de Comportamiento:")
print("- dia_semana_favorito, hora_favorita, mes_mas_activo")

print("\nVariables Geográficas:")
print("- pais_principal, paises_diferentes")

print("\nVariables de Diversidad:")
print("- categorias_productos_diferentes, categoria_favorita, diversidad_categorias")

print("\nVariables Mensuales:")
print("- meses_activo, valor_mensual_medio, compras_mensuales_medias")


print("\n=== 8. GUARDADO DEL DATASET FINAL ===")
# Guardar el dataset centrado en clientes
ruta_salida = "./DataSet/dataset_clientes_clustering.csv"
df_clientes.to_csv(ruta_salida, index=False, encoding="latin-1")
print(f"Dataset guardado en: {ruta_salida}")

print(f"\n=== RESUMEN FINAL ===")
print(f"Dataset original: {df.shape[0]:,} transacciones")
print(f"Dataset clientes: {df_clientes.shape[0]:,} clientes únicos") 
print(f"Variables por cliente: {df_clientes.shape[1]:,} características")
print(f"Periodo analizado: {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}")

# Mostrar estadísticas básicas del dataset de clientes
print(f"\n=== ESTADÍSTICAS BÁSICAS DEL DATASET DE CLIENTES ===")
print(df_clientes.describe().round(2))

