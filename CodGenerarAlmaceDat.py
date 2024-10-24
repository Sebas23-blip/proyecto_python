import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

# Función para resolver el circuito LC (Movimiento Armónico Simple)
def lc_system(y, t, L, C):
    q, i = y  # Carga y corriente
    dydt = [i, -q / (L * C)]  # Sistema de ecuaciones diferenciales
    return dydt

# Función para resolver el circuito RLC Amortiguado
def rlc_system(y, t, R, L, C):
    q, i = y  # Carga y corriente
    dydt = [i, -R / L * i - q / (L * C)]  # Ecuaciones diferenciales de RLC amortiguado
    return dydt

# Función para resolver el circuito RLC Amortiguado Forzado
def rlc_forzado_system(y, t, R, L, C, V0, omega):
    q, i = y  # Carga y corriente
    dydt = [i, -R / L * i - q / (L * C) + V0 * np.cos(omega * t) / L]  # Ecuaciones de RLC forzado
    return dydt

# Función para generar datos y simular el circuito
def simular_circuito(tipo, L, C, R=0, V0=0, omega=0, t_max=10, num_points=1000, file_name=None):
    t = np.linspace(0, t_max, num_points)  # Tiempo de simulación

    if tipo == "LC":
        y0 = [1, 0]  # Condiciones iniciales: q(0) = 1, i(0) = 0
        sol = odeint(lc_system, y0, t, args=(L, C))
        label = "Movimiento Armónico Simple (LC)"

    elif tipo == "RLC_amortiguado":
        y0 = [1, 0]  # Condiciones iniciales: q(0) = 1, i(0) = 0
        sol = odeint(rlc_system, y0, t, args=(R, L, C))
        label = "Movimiento Armónico Amortiguado (RLC)"

    elif tipo == "RLC_forzado":
        y0 = [0, 0]  # Condiciones iniciales: q(0) = 0, i(0) = 0
        sol = odeint(rlc_forzado_system, y0, t, args=(R, L, C, V0, omega))
        label = "Movimiento Armónico Forzado (RLC)"

    q = sol[:, 0]  # Carga (q)
    i = sol[:, 1]  # Corriente (i)

    # Graficar los resultados (opcional)
    plt.figure(figsize=(10, 6))
    plt.plot(t, q, label='Carga (q)', color='blue')
    plt.plot(t, i, label='Corriente (i)', color='red')
    plt.title(f'Simulación: {label}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Guardar los datos en un archivo CSV
    if file_name:
        # Ruta para guardar el archivo
        directorio = r"C:\Users\User\PycharmProjects\ProyectoLc\directorio"
        if not os.path.exists(directorio):
            os.makedirs(directorio)  # Crea la carpeta si no existe
        file_path = os.path.join(directorio, file_name)
        datos = np.column_stack((t, q, i))
        df = pd.DataFrame(datos, columns=['Tiempo', 'Carga (q)', 'Corriente (i)'])
        df.to_csv(file_path, index=False)
        print(f"Datos guardados en {file_path}")

# Función para generar configuraciones aleatorias de los parámetros
def generar_configuracion_aleatoria(tipo_circuito):
    # Rango de valores para L (Henrys), C (Faradios), R (Ohmios), V0 (Volts), y omega (rad/s)
    L = random.uniform(0.01, 10)  # Inductancia entre 0.01 H y 10 H
    C = random.uniform(0.001, 0.1)  # Capacitancia entre 0.001 F y 0.1 F

    if tipo_circuito == "LC":
        return L, C, None, None, None  # No se necesita R, V0 ni omega

    elif tipo_circuito == "RLC_amortiguado":
        R = random.uniform(0.1, 100)  # Resistencia entre 0.1 Ω y 100 Ω
        return L, C, R, None, None  # No se necesita V0 ni omega

    elif tipo_circuito == "RLC_forzado":
        R = random.uniform(0.1, 100)  # Resistencia entre 0.1 Ω y 100 Ω
        V0 = random.uniform(1, 10)  # Voltaje externo entre 1 V y 10 V
        omega = random.uniform(0.1, 10)  # Frecuencia angular entre 0.1 y 10 rad/s
        return L, C, R, V0, omega

# Función para ejecutar varias simulaciones aleatorias
def ejecutar_simulaciones_aleatorias(num_simulaciones, tipo_circuito):
    for i in range(num_simulaciones):
        # Generar parámetros aleatorios
        configuracion = generar_configuracion_aleatoria(tipo_circuito)
        L, C, R, V0, omega = configuracion

        # Asignar nombre al archivo de salida
        file_name = f"datos_{tipo_circuito}_simulacion_{i + 1}.csv"

        # Ejecutar la simulación con los parámetros aleatorios generados
        if tipo_circuito == "LC":
            simular_circuito("LC", L, C, file_name=file_name)
        elif tipo_circuito == "RLC_amortiguado":
            simular_circuito("RLC_amortiguado", L, C, R, file_name=file_name)
        elif tipo_circuito == "RLC_forzado":
            simular_circuito("RLC_forzado", L, C, R, V0, omega, file_name=file_name)

# Función principal para seleccionar el tipo de circuito y ejecutar simulaciones aleatorias
def main():
    print("Selecciona el tipo de circuito para generar simulaciones aleatorias:")
    print("1. Movimiento Armónico Simple (LC)")
    print("2. Movimiento Armónico Amortiguado (RLC)")
    print("3. Movimiento Armónico Forzado (RLC)")
    tipo_circuito = int(input("Ingrese el número correspondiente: "))

    if tipo_circuito == 1:
        tipo = "LC"
    elif tipo_circuito == 2:
        tipo = "RLC_amortiguado"
    elif tipo_circuito == 3:
        tipo = "RLC_forzado"
    else:
        print("Opción no válida.")
        return

    num_simulaciones = int(input("Ingrese el número de simulaciones a realizar: "))

    # Ejecutar las simulaciones aleatorias
    ejecutar_simulaciones_aleatorias(num_simulaciones, tipo)

if __name__ == "__main__":
    main()
