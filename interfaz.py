import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


class LCPredictor:
    def __init__(self):
        """
        Inicializa el predictor para el rango de 10 mHz a 10 Hz
        """
        self.freq_min = 0.01  # 10 mHz
        self.freq_max = 10.0  # 10 Hz
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    def generate_training_data(self, n_samples=1000):
        """Genera datos de entrenamiento en el rango de bajas frecuencias"""
        # Usar distribución logarítmica para mejor cobertura del rango
        frequencies = np.logspace(
            np.log10(self.freq_min),
            np.log10(self.freq_max),
            n_samples
        )

        # Cálculo de L y C adaptado para bajas frecuencias
        # Estos valores son ajustados para el rango de frecuencias bajo
        L = 1 / (4 * np.pi * np.pi * frequencies * frequencies * 1e-6)  # Valores más grandes de L
        C = 1e-6 * np.ones_like(frequencies)  # Valores más grandes de C

        return frequencies, L, C

    def prepare_data(self):
        """Prepara los datos para el entrenamiento"""
        frequencies, L, C = self.generate_training_data()
        X = frequencies.reshape(-1, 1)
        y = np.column_stack((L, C))

        # Aplicar transformación logarítmica
        X = np.log10(X)
        y = np.log10(y)  # También transformamos L y C

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def create_model(self):
        """Crea el modelo de red neuronal adaptado para bajas frecuencias"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, epochs=100, batch_size=32):
        """Entrena el modelo"""
        print(f"\nEntrenando modelo para rango de frecuencia: {self.freq_min * 1000:.1f} mHz - {self.freq_max:.1f} Hz")
        X_train, X_test, y_train, y_test = self.prepare_data()

        self.model = self.create_model()
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        return history

    def predict(self, frequency):
        """Realiza predicción para una frecuencia dada"""
        if frequency < self.freq_min or frequency > self.freq_max:
            print(f"\n⚠️ Advertencia: La frecuencia está fuera del rango de entrenamiento "
                  f"({self.freq_min * 1000:.1f} mHz - {self.freq_max:.1f} Hz)")

        freq_log = np.log10(frequency).reshape(1, -1)
        freq_scaled = self.scaler_X.transform(freq_log)
        prediction_scaled = self.model.predict(freq_scaled)
        prediction_log = self.scaler_y.inverse_transform(prediction_scaled)
        prediction = 10 ** prediction_log  # Revertir transformación logarítmica
        return prediction[0]


def get_user_frequency():
    """Obtiene la frecuencia del usuario en mHz o Hz"""
    while True:
        try:
            print("\nIngrese la frecuencia:")
            print("1. milihertz (mHz)")
            print("2. hertz (Hz)")

            option = int(input("Seleccione una unidad (1-2): "))
            if option not in [1, 2]:
                print("Por favor, seleccione una opción válida (1-2)")
                continue

            value = float(input("Ingrese el valor numérico: "))
            if value <= 0:
                print("Por favor, ingrese un valor positivo")
                continue

            # Convertir a Hz
            if option == 1:  # mHz
                return value / 1000
            else:  # Hz
                return value

        except ValueError:
            print("Por favor, ingrese un número válido")


def format_frequency(freq_hz):
    """Formatea la frecuencia en mHz o Hz según corresponda"""
    if freq_hz < 1:
        return f"{freq_hz * 1000:.3f} mHz"
    else:
        return f"{freq_hz:.3f} Hz"


def main():
    print("Predictor de valores L y C para Bajas Frecuencias")
    print("-----------------------------------------------")
    print(f"Rango de operación: 10 mHz - 10 Hz")

    # Crear y entrenar el modelo
    predictor = LCPredictor()
    history = predictor.train()

    # Visualizar el entrenamiento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Evolución del entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

    # Graficar predicciones en todo el rango
    test_frequencies = np.logspace(np.log10(0.01), np.log10(10), 100)
    L_values = []
    C_values = []

    for freq in test_frequencies:
        L, C = predictor.predict(freq)
        L_values.append(L)
        C_values.append(C)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogx(test_frequencies, L_values)
    plt.title('Valores de L vs Frecuencia')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Inductancia (H)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogx(test_frequencies, C_values)
    plt.title('Valores de C vs Frecuencia')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Capacitancia (F)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Loop de predicción
    while True:
        frequency = get_user_frequency()
        L_pred, C_pred = predictor.predict(frequency)

        print("\nResultados de la predicción:")
        print(f"Frecuencia: {format_frequency(frequency)}")
        print(f"L predicho: {L_pred:.3e} H")
        print(f"C predicho: {C_pred:.3e} F")

        if input("\n¿Desea hacer otra predicción? (s/n): ").lower() != 's':
            break

    print("\n¡Gracias por usar el predictor de valores L y C!")


if __name__ == "__main__":
    main()