import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Función para calcular la frecuencia de resonancia de un circuito RLC
def frecuencia_resonancia(L, C):
    return 1 / (2 * np.pi * np.sqrt(L * C))


# Función objetivo que mide la diferencia entre la frecuencia objetivo y la calculada
def funcion_objetivo(params, f_objetivo):
    L, C = params
    f_calculada = frecuencia_resonancia(L, C)
    error = (f_calculada - f_objetivo) ** 2  # Error cuadrático
    return error


# Optimización para obtener los valores de L y C
def optimizar_circuito(f_objetivo, L_inicial, C_inicial):
    # Definir los límites de los parámetros para la optimización
    limites = [(0.01, 10), (0.001, 1)]  # Límites para L y C

    # Ejecutar la optimización
    resultado = minimize(funcion_objetivo, [L_inicial, C_inicial], args=(f_objetivo,), bounds=limites)

    # Parámetros optimizados
    L_opt, C_opt = resultado.x
    return L_opt, C_opt


# Función principal para el registro de datos y optimización
def main():
    # Registro de la frecuencia objetivo por parte del usuario
    f_objetivo = float(input("Ingresa la frecuencia de resonancia deseada (Hz): "))

    # Registro de valores iniciales de L y C
    L_inicial = float(input("Ingresa el valor inicial de la inductancia L (Henrios): "))
    C_inicial = float(input("Ingresa el valor inicial de la capacitancia C (Faradios): "))

    # Optimizar los valores de L y C para alcanzar la frecuencia deseada
    L_opt, C_opt = optimizar_circuito(f_objetivo, L_inicial, C_inicial)

    print(f"Parámetros optimizados para alcanzar la frecuencia de resonancia de {f_objetivo} Hz:")
    print(f"Inductancia (L): {L_opt:.6f} H")
    print(f"Capacitancia (C): {C_opt:.6f} F")

    # Calcular la frecuencia obtenida con los parámetros optimizados
    f_obtenida = frecuencia_resonancia(L_opt, C_opt)
    print(f"Frecuencia obtenida con los parámetros optimizados: {f_obtenida:.6f} Hz")

    # Graficar los resultados
    etiquetas = ['Inductancia (L)', 'Capacitancia (C)', 'Frecuencia (f)']
    valores_usuario = [L_inicial, C_inicial, f_objetivo]
    valores_optimizados = [L_opt, C_opt, f_obtenida]

    plt.figure(figsize=(10, 6))

    # Graficar los valores iniciales y optimizados
    bar_width = 0.35
    indices = np.arange(len(etiquetas))

    plt.bar(indices, valores_usuario, bar_width, label='Valores Iniciales', color='blue')
    plt.bar(indices + bar_width, valores_optimizados, bar_width, label='Valores Optimizados', color='green')

    plt.xlabel('Parámetros')
    plt.ylabel('Valores')
    plt.title('Comparación de los Parámetros Iniciales y Optimizados')
    plt.xticks(indices + bar_width / 2, etiquetas)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
