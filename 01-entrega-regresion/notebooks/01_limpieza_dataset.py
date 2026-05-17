import pandas as pd
import holidays


RUTA_ENTRADA = r"C:\Users\ruizp\Downloads\data.csv\data.csv"
RUTA_SALIDA = r"C:\Users\ruizp\Documents\New project\dataset_original_limpio_es.csv"


def formato_es(numero):
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def calcular_limites_iqr(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return limite_inferior, limite_superior


def clasificar_producto(descripcion):
    descripcion = str(descripcion).lower()

    if "christmas" in descripcion:
        return "navidad"
    elif "heart" in descripcion or "love" in descripcion or "valentine" in descripcion:
        return "romantico"
    elif "baby" in descripcion:
        return "bebe"
    elif "birthday" in descripcion:
        return "cumpleanos"
    elif "postage" in descripcion or "manual" in descripcion or "discount" in descripcion:
        return "no_producto"
    else:
        return "otros"


mapa_paises = {
    "United Kingdom": "GB",
    "France": "FR",
    "Germany": "DE",
    "Spain": "ES",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Switzerland": "CH",
    "Portugal": "PT",
    "Italy": "IT",
    "EIRE": "IE",
    "Australia": "AU",
    "Norway": "NO",
    "Sweden": "SE",
    "Japan": "JP",
}

cache_festivos = {}


def es_festivo_por_pais(fecha, pais):
    codigo = mapa_paises.get(pais)

    if pd.isna(fecha) or codigo is None:
        return False

    anio = fecha.year
    clave = (codigo, anio)

    if clave not in cache_festivos:
        cache_festivos[clave] = holidays.country_holidays(codigo, years=anio)

    return fecha in cache_festivos[clave]


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


print("\n=== 4. VARIABLES DE FECHA ===")
df["fecha"] = pd.to_datetime(df["fecha_factura_original"], errors="coerce").dt.normalize()
df = df.dropna(subset=["fecha"]).copy()

mapa_dias = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}

df["dia_semana"] = df["fecha"].dt.day_name().map(mapa_dias)


df["dia_semana"] = df["fecha"].dt.day_name().map(mapa_dias)
df["es_festivo_pais"] = df.apply(
    lambda fila: es_festivo_por_pais(fila["fecha"], fila["pais"]),
    axis=1
)


print("\n=== 5. TRATAMIENTO DE LA DESCRIPCION PARA ANALISIS ===")
df["descripcion_limpia"] = (
    df["descripcion_original"]
    .fillna("sin_descripcion")
    .astype(str)
    .str.lower()
    .str.strip()
)

df["categoria_producto"] = df["descripcion_limpia"].apply(clasificar_producto)
df["flag_no_producto"] = df["categoria_producto"] == "no_producto"


print("\n=== 6. OUTLIERS CON REGLA 1,5 IQR ===")
lim_inf_cantidad, lim_sup_cantidad = calcular_limites_iqr(df["cantidad"])
lim_inf_precio, lim_sup_precio = calcular_limites_iqr(df["precio_unitario"])

df["flag_outlier_cantidad"] = (
    (df["cantidad"] < lim_inf_cantidad) | (df["cantidad"] > lim_sup_cantidad)
)
df["flag_outlier_precio"] = (
    (df["precio_unitario"] < lim_inf_precio) | (df["precio_unitario"] > lim_sup_precio)
)

df["importe_linea"] = df["cantidad"] * df["precio_unitario"]
lim_inf_importe, lim_sup_importe = calcular_limites_iqr(df["importe_linea"])
df["flag_outlier_importe"] = (
    (df["importe_linea"] < lim_inf_importe)
    | (df["importe_linea"] > lim_sup_importe)
)

print("Outliers en cantidad:", int(df["flag_outlier_cantidad"].sum()))
print("Outliers en precio:", int(df["flag_outlier_precio"].sum()))
print("Outliers en importe:", int(df["flag_outlier_importe"].sum()))


print("\n=== 7. IMPORTE TOTAL DE LA VENTA ===")
df["importe_total_venta"] = df["importe_linea"].map(formato_es)


print("\n=== 8. LIMPIEZA FINAL PARA POWER BI ===")
df = df.drop(columns=["fecha_factura_original"])


print("\n=== 9. EXPORTACION ===")
df.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
print("Archivo guardado en:", RUTA_SALIDA)
print("Filas finales:", len(df))
