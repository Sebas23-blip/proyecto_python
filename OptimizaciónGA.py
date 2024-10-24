import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


# Función que simula el circuito RLC
def rlc_system(y, t, R, L, C):
    q, i = y  # Carga y corriente
    dydt = [i, -R / L * i - q / (L * C)]  # Ecuaciones diferenciales de RLC amortiguado
    return dydt


# Función para simular el circuito RLC con parámetros dados
def simular_rlc(L, C, R, t_max=10, num_points=1000):
    t = np.linspace(0, t_max, num_points)
    y0 = [1, 0]  # Condiciones iniciales: q(0) = 1, i(0) = 0
    sol = odeint(rlc_system, y0, t, args=(R, L, C))
    q = sol[:, 0]  # Carga (q)
    i = sol[:, 1]  # Corriente (i)
    return t, q, i


# Función para calcular el factor de amortiguamiento zeta
def factor_amortiguamiento(R, L, C):
    return R / (2 * np.sqrt(L / C))


# Función objetivo que minimiza el factor de amortiguamiento zeta
def funcion_objetivo(params, R):
    L, C = params
    zeta = factor_amortiguamiento(R, L, C)
    return zeta


# Optimizar L y C para minimizar el factor de amortiguamiento zeta
def optimizar_rlc_minimizar_zeta(R):
    # Valores iniciales de L y C
    L_inicial = 1.0
    C_inicial = 0.01

    # Definir los límites de los parámetros para la optimización
    limites = [(0.01, 10), (0.001, 1)]  # Límites para L y C

    # Ejecutar la optimización
    resultado = minimize(funcion_objetivo, [L_inicial, C_inicial], args=(R,), bounds=limites)

    # Parámetros optimizados
    L_opt, C_opt = resultado.x
    return L_opt, C_opt


# Función principal para registro de datos y optimización
def main():
    # Registro de valores por parte del usuario
    R_usuario = float(input("Ingresa el valor de la resistencia R (Ohms): "))
    t_max = float(input("Ingresa el tiempo máximo de simulación (s): "))
    num_points = int(input("Ingresa el número de puntos de simulación: "))

    # Simular el circuito con los valores iniciales ingresados por el usuario
    L_usuario = float(input("Ingresa el valor de la inductancia L (Henrios): "))
    C_usuario = float(input("Ingresa el valor de la capacitancia C (Faradios): "))

    t_usuario, q_usuario, i_usuario = simular_rlc(L_usuario, C_usuario, R_usuario, t_max=t_max, num_points=num_points)

    # Optimizar los valores de L y C para minimizar el factor de amortiguamiento
    L_opt, C_opt = optimizar_rlc_minimizar_zeta(R_usuario)

    print(f"Parámetros optimizados para minimizar el factor de amortiguamiento: L = {L_opt:.4f} H, C = {C_opt:.4f} F")

    # Simular el circuito con los parámetros optimizados
    t_opt, q_opt, i_opt = simular_rlc(L_opt, C_opt, R_usuario, t_max=t_max, num_points=num_points)

    # Graficar los resultados
    plt.figure(figsize=(10, 6))

    # Gráfica de la carga y corriente ingresada por el usuario
    plt.plot(t_usuario, q_usuario, label='Carga Usuario (q)', color='blue')
    plt.plot(t_usuario, i_usuario, label='Corriente Usuario (i)', color='green')

    # Gráfica de la carga y corriente optimizadas
    plt.plot(t_opt, q_opt, label='Carga Optimizada (q)', color='red', linestyle='--')
    plt.plot(t_opt, i_opt, label='Corriente Optimizada (i)', color='orange', linestyle='--')

    plt.title('Optimización del Factor de Amortiguamiento en un Circuito RLC Amortiguado')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
