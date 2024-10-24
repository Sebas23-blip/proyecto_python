import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import tkinter as tk
from tkinter import messagebox


# Función que describe el sistema LC
def circuito_lc(y, t, L, C):
    q, dqdt = y
    dydt = [dqdt, -q / (L * C)]
    return dydt


# Función para realizar la simulación y graficar
def simular_lc():
    try:
        # Obtenemos los valores ingresados por el usuario
        L = float(entry_L.get())
        C = float(entry_C.get())

        # Parámetros iniciales
        q0 = 1.0  # Carga inicial
        dq0 = 0.0  # Corriente inicial

        # Tiempo para la simulación
        t = np.linspace(0, 10, 1000)

        # Solucionamos la ecuación diferencial
        solucion = odeint(circuito_lc, [q0, dq0], t, args=(L, C))
        q = solucion[:, 0]  # Carga
        dqdt = solucion[:, 1]  # Corriente (derivada de la carga)

        # Calcular voltaje en el capacitor
        V = q / C

        # Normalizamos la corriente para que tenga la misma amplitud que la carga
        dqdt_normalized = dqdt / np.max(np.abs(dqdt)) * np.max(np.abs(q))

        # Graficamos todas las curvas en una sola gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(t, q, label='Carga (q)', color='blue')
        plt.plot(t, dqdt_normalized, label='Corriente (dq/dt)', color='red')
        plt.plot(t, V, label='Voltaje (V)', color='green')
        plt.title(f'Simulación Circuito LC\nL = {L} H, C = {C} F')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos para L y C.")


# Crear la ventana principal
root = tk.Tk()
root.title("Simulación Circuito LC")

# Etiquetas y entradas de texto
tk.Label(root, text="Inductancia (L) [H]").grid(row=0)
tk.Label(root, text="Capacitancia (C) [F]").grid(row=1)

entry_L = tk.Entry(root)
entry_C = tk.Entry(root)

entry_L.grid(row=0, column=1)
entry_C.grid(row=1, column=1)

# Botón para iniciar la simulación
boton_simular = tk.Button(root, text="Simular", command=simular_lc)
boton_simular.grid(row=2, columnspan=2)
# Iniciar la aplicación
root.mainloop()
