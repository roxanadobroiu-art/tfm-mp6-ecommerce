import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


RUTA = r"C:\Users\ruizp\Documents\New project\dataset_original_limpio_es.csv"

df = pd.read_csv(RUTA)

# Convertir fecha
df["fecha"] = pd.to_datetime(df["fecha"])

# 2. SERIE TEMPORAL


serie = df.groupby("fecha")["importe_linea"].sum().sort_index()


train = serie[:'2011-11-08']
test = serie['2011-11-09':]

modelo = ARIMA(train, order=(1,1,1))
modelo_fit = modelo.fit()

predicciones = modelo_fit.forecast(steps=len(test))


# evaluamos

rmse = np.sqrt(mean_squared_error(test, predicciones))
print("RMSE:", rmse) ##RMSE = error medio de predicción


# GRAFICA

plt.figure(figsize=(12,5))
plt.plot(train, label="Train")
plt.plot(test, label="Real")
plt.plot(test.index, predicciones, label="Predicción")
plt.legend()
plt.show()