import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simulación: Generar datos de ejemplo
def simulate_rlc(R, L, C, t):
    # Simula el circuito RLC en serie para generar el comportamiento de la corriente
    # Este sería un conjunto de datos generado de la simulación real, simplificado aquí
    freq = 1 / np.sqrt(L * C)
    damping = R / (2 * L)
    current = np.exp(-damping * t) * np.cos(freq * t)
    return current

# Generar datos de entrenamiento
R_values = np.random.uniform(1, 100, 1000)  # Generar 1000 valores de R
L_values = np.random.uniform(0.1, 10, 1000) # Generar 1000 valores de L
C_values = np.random.uniform(0.01, 1, 1000) # Generar 1000 valores de C
t = np.linspace(0, 10, 100)  # Intervalo de tiempo

# Matriz de características
X = np.vstack((R_values, L_values, C_values)).T
# Etiquetas (resultado de la simulación)
y = np.array([simulate_rlc(R, L, C, t) for R, L, C in zip(R_values, L_values, C_values)])

# Crear el modelo de red neuronal
model = Sequential([
    Dense(64, input_dim=3, activation='relu'),
    Dense(128, activation='relu'),
    Dense(100, activation='linear')  # Salida con 100 puntos de corriente simulada
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X, y, epochs=50, batch_size=32)

# Predecir el comportamiento de un nuevo circuito con IA
R_test, L_test, C_test = 50, 2.0, 0.1  # Valores de prueba
prediction = model.predict(np.array([[R_test, L_test, C_test]]))
print("Predicción de la corriente:", prediction)
