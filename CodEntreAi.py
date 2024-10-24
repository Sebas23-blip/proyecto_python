import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar los datos del archivo CSV
df = pd.read_csv("rlc_data.csv")

# Separar las características (R, L, C, Time) y las etiquetas (Voltage)
X = df[['R', 'L', 'C', 'Time']].values
y = df['Voltage'].values

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = Sequential([
    Dense(64, input_dim=4, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')  # Salida para predecir el voltaje
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluar el modelo en los datos de prueba
loss = model.evaluate(X_test, y_test)
print(f"Error cuadrático medio en los datos de prueba: {loss}")

# Hacer predicciones con el modelo entrenado
predictions = model.predict(X_test)
# Imprimir las primeras 10 predicciones junto con los valores reales
for i in range(10):
    print(f"Valor real: {y_test[i]}, Predicción: {predictions[i][0]}")
