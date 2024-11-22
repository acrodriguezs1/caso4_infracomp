import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

import matplotlib.pyplot as plt
# Leer el archivo CSV
df = pd.read_csv('pruebas30k.csv')

# Extraer la columna 'timestamp'
timestamps = df['timeStamp']
grpThreads = df['grpThreads']

# Imprimir los timestamps
print(list(timestamps))
# Datos de ejemplo
X = np.array(list(timestamps))
y = np.array(list(grpThreads))

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# Visualizar los resultados
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal')
plt.legend()
plt.show()


