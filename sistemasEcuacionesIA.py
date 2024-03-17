import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Definir el sistema de ecuaciones lineales
A = np.array([[2, 3], [1, 2]])
b = np.array([[8], [5]])

# Definir el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=(2,)),  # Capa de entrada con 2 unidades para 2 variables
    tf.keras.layers.Dense(units=2),  # Capa oculta con 2 unidades
    tf.keras.layers.Dense(units=1)   # Capa de salida con 1 unidad
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con el sistema de ecuaciones
historial = modelo.fit(A, b, epochs=1000, verbose=False)
print("Modelo entrenado.")

# Graficar la pérdida durante el entrenamiento
plt.plot(historial.history['loss'])
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el entrenamiento')
plt.show()

# Evaluar el modelo
solucion = modelo.predict(A)

print("La solución del sistema de ecuaciones es:")
print(solucion)