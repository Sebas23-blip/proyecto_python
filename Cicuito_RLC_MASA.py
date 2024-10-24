import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk

# Definir la función diferencial que describe el comportamiento del circuito RLC
def rlc_system(y, t, R, L, C):
    i, v = y
    dydt = [v, -(R/L)*v - (1/(L*C))*i]
    return dydt

# Función para realizar la simulación y graficar los resultados
def simulate():
    try:
        R = float(entry_R.get())
        L = float(entry_L.get())
        C = float(entry_C.get())
        I0 = float(entry_I0.get())
        V0 = float(entry_V0.get())
        t_max = float(entry_tmax.get())
    except ValueError:
        result_label.config(text="Por favor, introduce valores numéricos válidos.")
        return

    t = np.linspace(0, t_max, 1000)  # Tiempo desde 0 hasta t_max
    y0 = [I0, V0]  # Condiciones iniciales: corriente y voltaje

    # Resolver las ecuaciones diferenciales
    sol = odeint(rlc_system, y0, t, args=(R, L, C))
    corriente = sol[:, 0]
    voltaje = sol[:, 1]

    # Crear las gráficas
    plt.figure(figsize=(10, 6))

    # Gráfica de corriente
    plt.subplot(2, 1, 1)
    plt.plot(t, corriente, label="Corriente (I)", color="blue")
    plt.title('Simulación de un circuito RLC - Corriente y Voltaje')
    plt.ylabel("Corriente (A)")
    plt.grid(True)

    # Gráfica de voltaje
    plt.subplot(2, 1, 2)
    plt.plot(t, voltaje, label="Voltaje (V)", color="red")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (V)")
    plt.grid(True)

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()

# Configuración de la ventana principal
root = tk.Tk()
root.title("Simulación de un circuito RLC")

# Etiquetas y entradas para los valores de R, L, C, I0, V0 y t_max
ttk.Label(root, text="Resistencia (R en ohmios):").grid(column=0, row=0, padx=10, pady=5)
entry_R = ttk.Entry(root)
entry_R.grid(column=1, row=0, padx=10, pady=5)

ttk.Label(root, text="Inductancia (L en henrios):").grid(column=0, row=1, padx=10, pady=5)
entry_L = ttk.Entry(root)
entry_L.grid(column=1, row=1, padx=10, pady=5)

ttk.Label(root, text="Capacitancia (C en faradios):").grid(column=0, row=2, padx=10, pady=5)
entry_C = ttk.Entry(root)
entry_C.grid(column=1, row=2, padx=10, pady=5)

ttk.Label(root, text="Corriente inicial (I0 en A):").grid(column=0, row=3, padx=10, pady=5)
entry_I0 = ttk.Entry(root)
entry_I0.grid(column=1, row=3, padx=10, pady=5)

ttk.Label(root, text="Voltaje inicial (V0 en V):").grid(column=0, row=4, padx=10, pady=5)
entry_V0 = ttk.Entry(root)
entry_V0.grid(column=1, row=4, padx=10, pady=5)

ttk.Label(root, text="Tiempo máximo (t_max en s):").grid(column=0, row=5, padx=10, pady=5)
entry_tmax = ttk.Entry(root)
entry_tmax.grid(column=1, row=5, padx=10, pady=5)

# Botón para ejecutar la simulación
btn_simulate = ttk.Button(root, text="Simular", command=simulate)
btn_simulate.grid(column=0, row=6, columnspan=2, pady=10)

# Etiqueta para mostrar mensajes
result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=7, columnspan=2)

# Iniciar el bucle de la interfaz gráfica
root.mainloop()
