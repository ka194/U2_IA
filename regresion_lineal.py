
import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de ejemplo (tamaño en metros cuadrados, precio en pesos mexicanos)
tamanios = np.array([50, 75, 100, 125, 150]).reshape((-1, 1))
precios_dolares = np.array([100000, 150000, 200000, 250000, 300000])

# Conversión de precios de dólares a pesos mexicanos (suponiendo que 1 USD = 18 MXN)
tipo_cambio = 18
precios_pesos = precios_dolares * tipo_cambio

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(tamanios, precios_pesos)

# Predicción del precio de una casa de 110 metros cuadrados
precio_predicho = modelo.predict(np.array([[110]]))
print(f"Precio predicho en pesos mexicanos: {precio_predicho[0]:,.2f}")