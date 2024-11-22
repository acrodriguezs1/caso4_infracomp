import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('pruebas30K.csv')

# Seleccionar las columnas relevantes
X = data[['allThreads']]
y = data['elapsed']

# Crear el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# Graficar los resultados
plt.scatter(X, y, color='pink', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Regresión lineal')
plt.xlabel('Timestamp')
plt.ylabel('grpThreads')
plt.title('Regresión Lineal')
plt.legend()
plt.show()