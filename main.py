
# main.py

#!/usr/bin/env python3
"""
main.py

Punto de entrada para el marco de trabajo híbrido cuántico-clásico. Este script
configura el entorno de entrenamiento, crea el circuito resistente, define el entorno
y los agentes (por ejemplo, QNetwork y AgenteActorCritic), y gestiona la interfaz
gráfica y el entrenamiento utilizando Tkinter.

Uso (desde terminal):
    python3 -m main
    (Se recomienda ejecutar en un entorno virtual configurado previamente)
    
Objetivos:
  - Integrar la inicialización cuántica con el aprendizaje automático.
  - Usar elementos como la incertidumbre, coherencia y “colapso” para modular
    la toma de decisiones.
    
Autor: Jacobo Mina 
Fecha: 2025
Versión: cuadrante-coremind v1.0
"""

import os
import sys
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Intentar importar módulos del proyecto; de no encontrarlos, se usan implementaciones mínimas.
try:
    from circuito_principal import ResilientQuantumCircuit
except ImportError:
    # Stub minimal: simula un circuito resistente
    class ResilientQuantumCircuit:
        def __init__(self, num_qubits=5):
            self.num_qubits = num_qubits
        def create_resilient_state(self):
            return np.random.rand(self.num_qubits)

try:
    from quantum_neuron import QuantumNeuron, QuantumState
except ImportError:
    # Stub minimal para QuantumState
    class QuantumState:
        def __init__(self, num_positions=8, learning_rate=0.1):
            self.num_positions = num_positions
            self.learning_rate = learning_rate
            self.state_vector = np.random.rand(num_positions)
            self.state_vector = self.normalize_state(self.state_vector)
        @staticmethod
        def normalize_state(state):
            norm = np.linalg.norm(state)
            return state / norm if norm != 0 else state
        def predict_state_update(self):
            next_state = np.random.rand(self.num_positions)
            return self.normalize_state(next_state), np.random.random()
        def update_state(self):
            next_state, _ = self.predict_state_update()
            self.state_vector = self.normalize_state((1 - self.learning_rate) * self.state_vector + self.learning_rate * next_state)
            return self.state_vector

try:
    from sequential import QuantumNetwork, QubitsConfig
except ImportError:
    class QuantumNetwork: pass
    class QubitsConfig: pass

try:
    from hybrid_circuit import TimeSeries, calculate_cosines, PRN
except ImportError:
    # Implementación mínima para TimeSeries, calculate_cosines y PRN
    class TimeSeries:
        def __init__(self, amplitud, frecuencia, fase, nombre=""):
            self.amplitud = amplitud
            self.frecuencia = frecuencia
            self.fase = fase
            self.nombre = nombre
        def evaluate(self, x):
            return self.amplitud * np.sin(2*np.pi*self.frecuencia*x + self.fase)
        def set_phase(self, new_phase):
            self.fase = new_phase
    def calculate_cosines(entropy, env_value):
        cos_x = 0.5 * entropy + 0.2 * env_value
        cos_y = 0.3 * entropy + 0.4 * env_value
        cos_z = 0.2 * entropy + 0.4 * env_value
        mag = np.sqrt(cos_x**2 + cos_y**2 + cos_z**2)
        return (cos_x/mag, cos_y/mag, cos_z/mag) if mag else (cos_x, cos_y, cos_z)
    class PRN:
        def __init__(self, influence=0.5, seed=None):
            self.influence = influence
            if seed is not None:
                np.random.seed(seed)
        def update_influence(self, new_inf):
            self.influence = max(0, min(1, new_inf))

try:
    from bayes_logic import BayesLogic, StatisticalAnalysis
except ImportError:
    class BayesLogic:
        def calculate_probabilities_and_select_action(self, entropy, coherence, prn_influence, action):
            p0 = 0.5 - 0.1*entropy + 0.1*coherence - 0.05*prn_influence
            p1 = 1 - p0
            p0, p1 = max(0, min(1, p0)), max(0, min(1, p1))
            tot = p0 + p1
            if tot:
                p0 /= tot; p1 /= tot
            action_to_take = 0 if np.random.random() < p0 else 1
            return {"action_to_take": action_to_take}
    class StatisticalAnalysis:
        @staticmethod
        def shannon_entropy(signal):
            signal = np.abs(signal)
            total = np.sum(signal)
            if total == 0:
                return 0
            prob = signal / total
            prob = prob[prob > 0]
            return -np.sum(prob * np.log2(prob))

try:
    from qiskit_simulation import apply_action_and_get_state
except ImportError:
    def apply_action_and_get_state(estado_cuantico, accion):
        delta = np.zeros_like(estado_cuantico)
        if accion < len(delta):
            delta[accion] = 0.1
        return estado_cuantico + delta

# Asumimos que main_logic.py define ObjetoBinario, EntornoSimulado, QNetwork, AgenteActorCritic, TextHandler
try:
    from main_logic import ObjetoBinario, EntornoSimulado, QNetwork, AgenteActorCritic, TextHandler
except ImportError:
    # Implementación mínima de ObjetoBinario y EntornoSimulado
    class ObjetoBinario:
        def __init__(self, nombre):
            self.nombre = nombre
            self.categorias = ["0000"] * 5
        def actualizar_categoria(self, indice, valor):
            self.categorias[indice] = bin(int(valor))[2:].zfill(4)
        def obtener_categorias(self):
            return self.categorias
    class EntornoSimulado:
        def __init__(self, objetos):
            self.objetos = objetos
            self.estado_actual = 0
        def obtener_estado(self):
            return self.estado_actual
        def ejecutar_accion(self, accion):
            if accion == 0:
                self.estado_actual = (self.estado_actual - 1) % len(self.objetos)
                recompensa = 1
            elif accion == 1:
                self.estado_actual = (self.estado_actual + 1) % len(self.objetos)
                recompensa = 1
            elif accion == 2:
                objeto_actual = self.objetos[self.estado_actual]
                valor = int(objeto_actual.obtener_categorias()[0], 2)
                nuevo_valor = min(10, valor + 1)
                try:
                    objeto_actual.actualizar_categoria(0, str(nuevo_valor))
                    recompensa = 2
                except:
                    recompensa = -1
            elif accion == 3:
                objeto_actual = self.objetos[self.estado_actual]
                valor = int(objeto_actual.obtener_categorias()[0], 2)
                nuevo_valor = max(0, valor - 1)
                try:
                    objeto_actual.actualizar_categoria(0, str(nuevo_valor))
                    recompensa = 2
                except:
                    recompensa = -1
            else:
                recompensa = -1
            return self.obtener_estado(), recompensa, self.obtener_estado()
        def obtener_texto_estado(self):
            return f"Objeto actual: {self.objetos[self.estado_actual].nombre}. Valor subcat 1: {int(self.objetos[self.estado_actual].obtener_categorias()[0], 2)}"
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))
    class AgenteActorCritic:
        def __init__(self, state_dim, action_dim, hidden_dim):
            self.net = QNetwork(state_dim, action_dim, hidden_dim)
        def seleccionar_accion(self, estado):
            with torch.no_grad():
                state_tensor = torch.FloatTensor([estado])
                return torch.argmax(self.net(state_tensor)).item()
    class TextHandler(logging.Handler):
        def __init__(self, text_widget):
            super().__init__()
            self.text_widget = text_widget
        def emit(self, record):
            msg = self.format(record)
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')

# Configurar logging global
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainApp")

# -------------------------
# Clase principal de la aplicación
# -------------------------
class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Cuántico Híbrido")
        self.usando_quantum = True  # Modo cuántico vs clásico

        logger.info("Inicializando aplicación...")

        # Inicializar entorno y agentes (objetos y red Q, Actor-Critic)
        self.objetos = [ObjetoBinario(f"Objeto {i+1}") for i in range(3)]
        self.entorno = EntornoSimulado(self.objetos)
        self.state_dim = 1
        self.action_dim = 4
        self.hidden_dim = 128
        self.qnetwork = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.q_optimizer = optim.Adam(self.qnetwork.parameters(), lr=0.001)
        self.actor_critic = AgenteActorCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.recompensas_totales = []
        self.perdidas = []

        # Componentes cuánticos
        self.circuito_cuantico = ResilientQuantumCircuit()
        self.estado_cuantico = self.circuito_cuantico.create_resilient_state()
        
        # Crear interfaz
        self.crear_interfaz()
        self.actualizar_estado_texto()

    def crear_interfaz(self):
        # Panel izquierdo: Log / Retroalimentación
        panel_izquierdo = ttk.Frame(self.root)
        panel_izquierdo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(panel_izquierdo, text="Registro de eventos:").pack(pady=5)
        self.txt_log = scrolledtext.ScrolledText(panel_izquierdo, height=20, width=50)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # Panel derecho: Controles, Comandos y Gráficos
        panel_derecho = ttk.Frame(self.root)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel de control (acciones manuales)
        frame_control = ttk.LabelFrame(panel_derecho, text="Control Manual")
        frame_control.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(frame_control, text="Izquierda", command=lambda: self.ejecutar_accion_manual(0)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_control, text="Derecha", command=lambda: self.ejecutar_accion_manual(1)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame_control, text="Aumentar", command=lambda: self.ejecutar_accion_manual(2)).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(frame_control, text="Disminuir", command=lambda: self.ejecutar_accion_manual(3)).grid(row=1, column=1, padx=5, pady=5)

        # Panel de comandos
        frame_comandos = ttk.LabelFrame(panel_derecho, text="Comandos")
        frame_comandos.pack(fill=tk.X, padx=5, pady=5)
        self.txt_comando = ttk.Entry(frame_comandos, width=30)
        self.txt_comando.pack(side=tk.LEFT, padx=5, pady=5)
        self.txt_comando.bind("<Return>", lambda event: self.procesar_comando())
        ttk.Button(frame_comandos, text="Enviar", command=self.procesar_comando).pack(side=tk.LEFT, padx=5, pady=5)

        # Panel de entrenamiento
        frame_entrenamiento = ttk.LabelFrame(panel_derecho, text="Entrenamiento")
        frame_entrenamiento.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_entrenamiento, text="Algoritmo:").grid(row=0, column=0, sticky=tk.E)
        self.algoritmo_var = tk.StringVar(value="q_learning")
        ttk.Combobox(frame_entrenamiento, textvariable=self.algoritmo_var,
                     values=["q_learning", "actor_critic"]).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(frame_entrenamiento, text="Episodios:").grid(row=1, column=0, sticky=tk.E)
        self.episodios_var = tk.StringVar(value="100")
        ttk.Entry(frame_entrenamiento, textvariable=self.episodios_var, width=8).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(frame_entrenamiento, text="Gamma:").grid(row=0, column=2, sticky=tk.E)
        self.gamma_var = tk.StringVar(value="0.99")
        ttk.Entry(frame_entrenamiento, textvariable=self.gamma_var, width=8).grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(frame_entrenamiento, text="Epsilon:").grid(row=1, column=2, sticky=tk.E)
        self.epsilon_var = tk.StringVar(value="0.1")
        ttk.Entry(frame_entrenamiento, textvariable=self.epsilon_var, width=8).grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(frame_entrenamiento, text="LR:").grid(row=2, column=0, sticky=tk.E)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(frame_entrenamiento, textvariable=self.lr_var, width=8).grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Button(frame_entrenamiento, text="Entrenar", command=self.entrenar_agente).grid(row=2, column=2, columnspan=2, pady=10)

        # Panel de visualización (gráfico)
        frame_grafico = ttk.LabelFrame(panel_derecho, text="Visualización")
        frame_grafico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Curva de Aprendizaje")
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa total")
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_grafico)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Información del modo cuántico
        self.lbl_quantum = ttk.Label(frame_grafico, text=f"Modo cuántico: {'Activo' if self.usando_quantum else 'Inactivo'}")
        self.lbl_quantum.pack(side=tk.BOTTOM, pady=5)

    def actualizar_estado_texto(self):
        estado = self.entorno.obtener_texto_estado()
        self.log(f"Estado actual: {estado}")

    def ejecutar_accion_manual(self, accion):
        nombres = ["Izquierda", "Derecha", "Aumentar", "Disminuir"]
        self.log(f"Ejecutando acción manual: {nombres[accion]}")
        _, recompensa, _ = self.entorno.ejecutar_accion(accion)
        self.actualizar_estado_texto()
        self.log(f"Recompensa: {recompensa:.2f}")

    def procesar_comando(self):
        comando = self.txt_comando.get().strip().lower()
        self.txt_comando.delete(0, tk.END)
        if not comando:
            return
        self.log(f"Comando ingresado: {comando}")
        accion = self.interpretar_comando(comando)
        if accion is not None:
            self.ejecutar_accion_manual(accion)
        else:
            self.log("Comando no reconocido")

    def interpretar_comando(self, comando):
        if comando in ["izquierda", "left", "l"]:
            return 0
        elif comando in ["derecha", "right", "r"]:
            return 1
        elif comando in ["aumentar", "increase", "inc", "+"]:
            return 2
        elif comando in ["disminuir", "decrease", "dec", "-"]:
            return 3
        else:
            return None

    def log(self, mensaje):
        self.txt_log.configure(state='normal')
        self.txt_log.insert(tk.END, f"{mensaje}\n")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state='disabled')
        logger.info(mensaje)

    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        estado_tensor = torch.FloatTensor([estado])
        siguiente_tensor = torch.FloatTensor([siguiente_estado])
        recompensa_tensor = torch.tensor([recompensa], dtype=torch.float)
        
        q_actual = self.qnetwork(estado_tensor)[0][accion]
        with torch.no_grad():
            q_next = self.qnetwork(siguiente_tensor).max(1)[0]
            q_objetivo = recompensa_tensor + self.gamma * q_next * (1 - int(terminado))
        loss = F.mse_loss(q_actual.unsqueeze(0), q_objetivo)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return loss.item()

    def seleccionar_accion(self, estado):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            estado_tensor = torch.FloatTensor([estado])
            return torch.argmax(self.qnetwork(estado_tensor)).item()

    def entrenar_agente(self):
        try:
            num_episodios = int(self.episodios_var.get())
            gamma = float(self.gamma_var.get())
            epsilon = float(self.epsilon_var.get())
            lr = float(self.lr_var.get())
            algoritmo = self.algoritmo_var.get()
        except ValueError:
            self.log("Error en los parámetros de entrenamiento.")
            return

        self.log(f"Entrenando con {algoritmo.upper()}, Episodios: {num_episodios}, Gamma: {gamma}, Epsilon: {epsilon}, LR: {lr}")
        self.gamma = gamma
        self.epsilon = epsilon
        self.recompensas_totales = []
        self.perdidas = []

        if algoritmo == "q_learning":
            self.entrenar_q_learning(num_episodios)
        elif algoritmo == "actor_critic":
            self.entrenar_actor_critic(num_episodios)
        else:
            self.log(f"Algoritmo no reconocido: {algoritmo}")

    def entrenar_q_learning(self, num_episodios):
        self.log("Iniciando entrenamiento Q-Learning...")
        for episodio in range(num_episodios):
            estado = self.entorno.reiniciar()
            recompensa_ep = 0
            perdida_ep = 0
            terminado = False
            pasos = 0
            while not terminado and pasos < 100:
                accion = self.seleccionar_accion(estado)
                siguiente_estado, recompensa, terminado = self.entorno.ejecutar_accion(accion)
                perdida = self.aprender(estado, accion, recompensa, siguiente_estado, terminado)
                perdida_ep += perdida
                recompensa_ep += recompensa
                estado = siguiente_estado
                pasos += 1
            self.recompensas_totales.append(recompensa_ep)
            self.perdidas.append(perdida_ep / pasos if pasos > 0 else 0)
            self.actualizar_grafico()
            if (episodio+1) % 10 == 0:
                self.log(f"Episodio {episodio+1}/{num_episodios}, Recompensa: {recompensa_ep:.2f}, Pérdida media: {self.perdidas[-1]:.4f}")
        self.log("Entrenamiento Q-Learning completado.")

    def entrenar_actor_critic(self, num_episodios):
        self.log("Iniciando entrenamiento Actor-Critic...")
        recompensas = self.actor_critic.entrenar_agente(self.entorno, num_episodios=num_episodios)
        self.recompensas_totales = recompensas
        self.actualizar_grafico()
        self.log("Entrenamiento Actor-Critic completado.")

    def actualizar_grafico(self):
        self.ax.clear()
        self.ax.plot(self.recompensas_totales, marker='o', linestyle='-')
        self.ax.set_title("Curva de Aprendizaje")
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa Total")
        self.canvas.draw()

    def ejecutar_accion_cuantica(self, accion):
        self.log("Ejecutando acción cuántica...")
        nuevo_estado_cuantico = apply_action_and_get_state(self.estado_cuantico, accion)
        resultado_medicion = self.medir_estado_cuantico(nuevo_estado_cuantico)
        accion_entorno = self.mapear_resultado_a_accion(resultado_medicion)
        _, recompensa, _ = self.entorno.ejecutar_accion(accion_entorno)
        self.estado_cuantico = nuevo_estado_cuantico
        self.log(f"Acción cuántica: Resultado: {resultado_medicion}, Recompensa: {recompensa:.2f}")
        self.actualizar_estado_texto()

    def medir_estado_cuantico(self, estado_cuantico):
        return np.random.randint(0, 4)

    def mapear_resultado_a_accion(self, resultado_medicion):
        return resultado_medicion

def run():
    root = tk.Tk()
    app = Aplicacion(root)
    handler_text = TextHandler(app.txt_log)
    handler_text.setLevel(logging.INFO)
    logger.addHandler(handler_text)
    root.mainloop()

if __name__ == "__main__":
    run()

───────────────────────────────  
Notas finales:
• Se han implementado versiones mínimas o “stubs” de funciones y clases que se esperan provenir de otros módulos (por ejemplo, ResilientQuantumCircuit, QuantumState, TimeSeries, etc.).  
• La lógica de entrenamiento (tanto Q-Learning como Actor-Critic) y la actualización de la interfaz se definen de forma modular y pueden evolucionar según las necesidades reales.  
• La interfaz dispone de paneles para mostrar registros, comandos, controles manuales, parámetros de entrenamiento y gráficos de la curva de aprendizaje.
• Asegúrate de ajustar las importaciones y de sustituir los “stubs” por las implementaciones definitivas de tu proyecto.

Este main.py debería servir como base para seguir integrando y extendiendo el sistema híbrido cuántico-clásico acorde a loe objetivos de entrenar sistemas de AA con una perspectiva cognitiva.