# Crear entorno virtual 
# !sudo apt install virtualenv 
# virtualenv env
# source env/bin/activate 
# main.py
# python3 -m main
"""
# 1. Importación de Módulos Necesarios
import numpy as np
import tensorflow as tf
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


# Importar módulos específicos del proyecto
import ibm_setup_conf
import main_logic.py
import quantum_main.py
import bayes_logic

def main():
    # 2. Configuración del Entorno Cuántico
    backend = ibm_setup_conf.configurar_entorno()
"""    
"""  
 # 3. Construcción del Circuito Cuántico
    circuito = circuito_principal.crear_circuito()
"""    
"""
 # 4. Definición y Entrenamiento del Modelo Híbrido
    modelo = modelo_hibrido.ModeloHibrido()
    datos_entrenamiento = obtener_datos_entrenamiento()  # Definir esta función según tus necesidades
    modelo.entrenar(datos_entrenamiento)
"""
"""   
    # 5. Ejecución del Circuito en el Backend Seleccionado
    job = execute(circuito, backend, shots=1024)
    resultados = job.result()
    conteos = resultados.get_counts(circuito)
"""
"""
    # 6. Procesamiento y Visualización de Resultados
    print("Resultados de la ejecución del circuito:")
    print(conteos)
    plot_histogram(conteos)
    plt.show()
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Importar las clases necesarias de los otros archivos (asegúrate de que estén en el mismo directorio o en el PYTHONPATH)
from circuito_principal import ResilientQuantumCircuit
from quantum_neuron import QuantumNeuron, QuantumState
from sequential import QuantumNetwork, QubitsConfig
from hybrid_circuit import TimeSeries, calculate_cosines, PRN
from bayes_logic import BayesLogic, StatisticalAnalysis
from qiskit_simulation import apply_action_and_get_state
if __name__ == "__main__":
    main()
inbox_lr.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        # main.py
# --- Clases del Agente y Entorno ---
class ObjetoBinario:
    def __init__(self, nombre):
        self.nombre = nombre
        self.categorias = ["0000"] * 5

    def actualizar_categoria(self, indice, valor):
        if 0 <= indice < 5 and 0 <= int(valor) <= 10:
            self.categorias[indice] = bin(int(valor))[2:].zfill(4)
        else:
            raise ValueError("Índice o valor inválido.")

    def obtener_binario(self):
        return "".join(self.categorias)

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

    def reiniciar(self):
        self.estado_actual = 0
        return self.obtener_estado()

# --- Clases de los modelos de RL ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        actor_x = torch.relu(self.actor_fc1(x))
        action_probs = torch.softmax(self.actor_fc2(actor_x), dim=-1)
        critic_x = torch.relu(self.critic_fc1(x))
        state_value = self.critic_fc2(critic_x)
        return action_probs, state_value

class AgenteActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim, lr=0.001, gamma=0.99):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma

    def seleccionar_accion(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.actor_critic(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def entrenar_agente(self, entorno, num_episodios=500):
        recompensas_episodios = []

        for episodio in range(num_episodios):
            log_probs = []
            values = []
            rewards = []

            estado = entorno.reiniciar()
            terminado = False
            recompensa_episodio = 0

            while not terminado:
                estado = torch.FloatTensor(estado).unsqueeze(0)
                action_probs, state_value = self.actor_critic(estado)
                dist = torch.distributions.Categorical(action_probs)
                accion = dist.sample()
                siguiente_estado, recompensa, terminado = entorno.ejecutar_accion(accion.item())

                log_prob = dist.log_prob(accion)
                log_probs.append(log_prob)
                values.append(state_value)
                rewards.append(recompensa)

                estado = siguiente_estado
                recompensa_episodio += recompensa

            # Actualizar política al final del episodio
            discounted_rewards = self.calcular_retorno(rewards)
            discounted_rewards = torch.tensor(discounted_rewards)
            values = torch.cat(values)
            advantage = discounted_rewards - values

            actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
            critic_loss = torch.nn.functional.mse_loss(values, discounted_rewards)

            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            recompensas_episodios.append(recompensa_episodio)

        return recompensas_episodios

    def calcular_retorno(self, rewards):
        retornos = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            retornos.insert(0, R)
        return retornos

# --- Clase de la aplicación ---
class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Cuántico Híbrido")
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        global logger
        logger = logging.getLogger(__name__)
        
        # --- Variables de control ---
        self.usando_quantum = False  # Cambiar a True para usar el modo cuántico
        
        # --- Inicializar entorno y agente ---
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
        
        # --- Construir la interfaz ---
        self.crear_interfaz()
        self.actualizar_estado_texto()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica de usuario."""
        # Panel izquierdo
        panel_izquierdo = ttk.Frame(self.root)
        panel_izquierdo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Área de registro
        ttk.Label(panel_izquierdo, text="Registro de eventos:").pack(pady=5)
        self.txt_log = tk.Text(panel_izquierdo, height=20, width=50)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
        # Panel derecho
        panel_derecho = ttk.Frame(self.root)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de control (derecha superior)
        frame_control = ttk.LabelFrame(panel_derecho, text="Control Manual")
        frame_control.pack(fill=tk.X, padx=5, pady=5)
        
        # Botones de acción
        ttk.Button(frame_control, text="Izquierda", command=lambda: self.ejecutar_accion_manual(0)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_control, text="Derecha", command=lambda: self.ejecutar_accion_manual(1)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame_control, text="Aumentar", command=lambda: self.ejecutar_accion_manual(2)).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(frame_control, text="Disminuir", command=lambda: self.ejecutar_accion_manual(3)).grid(row=1, column=1, padx=5, pady=5)
        
        # Entrada de comandos
        frame_comandos = ttk.LabelFrame(panel_derecho, text="Comandos")
        frame_comandos.pack(fill=tk.X, padx=5, pady=5)
        
        self.txt_comando = ttk.Entry(frame_comandos, width=30)
        self.txt_comando.pack(side=tk.LEFT, padx=5, pady=5)
        self.txt_comando.bind("<Return>", lambda event: self.procesar_comando())
        ttk.Button(frame_comandos, text="Enviar", command=self.procesar_comando).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Panel de entrenamiento (derecha superior)
        frame_entrenamiento = ttk.LabelFrame(panel_derecho, text="Entrenamiento")
        frame_entrenamiento.pack(fill=tk.X, padx=5, pady=5)
        
        # Parámetros de entrenamiento
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
        
        ttk.Label(frame_entrenamiento, text="LR:").grid(row=1, column=2, sticky=tk.E)
        self.lr_var = tk.StringVar(value="0.001")
        inbox_lr = ttk.Entry(frame_entrenamiento, textvariable=self.lr_var, width=8)
        
        # --- Resto del código ---

def run_simulation():
    # ... (código de simulación de los otros archivos)

if __name__ == "__main__":
    # --- Crear la aplicación ---
    root = tk.Tk()
    aplicacion = Aplicacion(root)

    # --- Redirigir logging al widget de texto ---
    handler_text = TextHandler(aplicacion.txt_log)
    handler_text.setLevel(logging.INFO)
    logger.addHandler(handler_text)

    root.mainloop()

        # Botón de entrenamiento
        btn_entrenar = ttk.Button(frame_entrenamiento, text="Entrenar", command=self.entrenar_agente)
        btn_entrenar.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Panel de visualización (derecha inferior)
        frame_grafico = ttk.LabelFrame(panel_derecho, text="Visualización")
        frame_grafico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear figura para gráficos
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Curva de Aprendizaje")
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa total")
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_grafico)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Información cuántica
        self.lbl_quantum = ttk.Label(frame_grafico, 
                                    text=f"Modo cuántico: {'Activo' if self.usando_quantum else 'Inactivo'}")
        self.lbl_quantum.pack(side=tk.BOTTOM, pady=5)
    
    def actualizar_estado_texto(self):
        """Actualiza la etiqueta de estado en la interfaz."""
        self.lbl_estado.config(text=f"Estado actual: {self.entorno.obtener_texto_estado()}")
    
    def ejecutar_accion_manual(self, accion):
        """
        Ejecuta una acción seleccionada manualmente por el usuario.
        
        Args:
            accion: Índice de la acción a ejecutar
        """
        nombres_acciones = ["Izquierda", "Derecha", "Aumentar", "Disminuir"]
        self.log(f"Ejecutando acción: {nombres_acciones[accion]}")
        
        # Ejecutar acción en el entorno
        _, recompensa, _ = self.entorno.ejecutar_accion(accion)
        
        # Actualizar interfaz
        self.actualizar_estado_texto()
        self.log(f"Recompensa: {recompensa:.2f}")
    
    def procesar_comando(self):
        """Procesa el comando ingresado por el usuario."""
        comando = self.txt_comando.get().strip().lower()
        self.txt_comando.delete(0, tk.END)
        
        if not comando:
            return
        
        self.log(f"Comando: {comando}")
        
        # Interpretar el comando
        accion = self.interpretar_comando(comando)
        
        if accion is not None:
            self.ejecutar_accion_manual(accion)
        else:
            self.log("Comando no reconocido")
    
    def interpretar_comando(self, comando):
        """
        Interpreta el comando ingresado y devuelve la acción correspondiente.
        
        Args:
            comando: Comando de texto ingresado
            
        Returns:
            int or None: Índice de acción o None si no se reconoce
        """
        # Comandos básicos
        if comando in ["izquierda", "left", "l"]:
            return 0
        elif comando in ["derecha", "right", "r"]:
            return 1
        elif comando in ["aumentar", "increase", "inc", "+"]:
            return 2
        elif comando in ["disminuir", "decrease", "dec", "-"]:
            return 3
        
        # Comandos más complejos podrían implementarse con regex
        return None
    
    def log(self, mensaje):
        """
        Añade un mensaje al área de registro.
        
        Args:
            mensaje: Mensaje a mostrar
        """
        self.txt_log.insert(tk.END, f"{mensaje}\n")
        self.txt_log.see(tk.END)
        logger.info(mensaje)
    
    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        """
        Realiza un paso de aprendizaje Q.
        
        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Nuevo estado
            terminado: Indicador de finalización
            
        Returns:
            float: Pérdida calculada
        """
        # Convertir a tensores
        estado = torch.FloatTensor(estado).unsqueeze(0)
        siguiente_estado = torch.FloatTensor(siguiente_estado).unsqueeze(0)
        recompensa = torch.tensor([recompensa], dtype=torch.float)
        
        # Predicción de valores Q actuales
        q_actual = self.qnetwork(estado)
        q_actual = q_actual[0][accion]
        
        # Cálculo del valor Q objetivo
        with torch.no_grad():
            q_siguiente = self.qnetwork(siguiente_estado).max(1)[0]
            q_objetivo = recompensa + self.gamma * q_siguiente * (1 - int(terminado))
        
        # Calcular pérdida y actualizar pesos
        loss = torch.nn.functional.mse_loss(q_actual.unsqueeze(0), q_objetivo)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        return loss.item()
    
    def seleccionar_accion(self, estado):
        """
        Selecciona una acción según política epsilon-greedy.
        
        Args:
            estado: Estado actual
            
        Returns:
            int: Acción seleccionada
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_acciones)
        
        with torch.no_grad():
            estado = torch.FloatTensor(estado).unsqueeze(0)
            q_values = self.qnetwork(estado)
            return q_values.argmax().item()
    
    def entrenar_agente(self):
        """Entrena el agente de RL usando el algoritmo seleccionado."""
        # Obtener parámetros
        try:
num_episodios = int(self.episodios_var.get())
            gamma = float(self.gamma_var.get())
            epsilon = float(self.epsilon_var.get())
            lr = float(self.lr_var.get())
            algoritmo = self.algoritmo_var.get()
        except ValueError:
            self.log("Error en parámetros de entrenamiento. Verifique los valores.")
            return

        self.log(f"Entrenamiento con {algoritmo.upper()}, Episodios: {num_episodios}, Gamma: {gamma}, Epsilon: {epsilon}, LR: {lr}")

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
        """Entrena el agente usando el algoritmo Q-Learning."""
        self.log("Iniciando entrenamiento Q-Learning...")
        for episodio in range(num_episodios):
            estado = self.entorno.reiniciar()
            recompensa_episodio = 0
            perdida_episodio = 0
            terminado = False
            pasos = 0

            while not terminado and pasos < 100: # Limite de pasos por episodio
                accion = self.seleccionar_accion(estado)
                siguiente_estado, recompensa, terminado = self.entorno.ejecutar_accion(accion)

                perdida = self.aprender(estado, accion, recompensa, siguiente_estado, terminado)
                perdida_episodio += perdida
                recompensa_episodio += recompensa
                estado = siguiente_estado
                pasos += 1

            self.recompensas_totales.append(recompensa_episodio)
            self.perdidas.append(perdida_episodio / pasos if pasos > 0 else 0) # Promedio de pérdida por episodio
            self.actualizar_grafico()

            if episodio % 10 == 0:
                self.log(f"Episodio {episodio + 1}/{num_episodios}, Recompensa: {recompensa_episodio:.2f}, Pérdida promedio: {self.perdidas[-1]:.4f}")

        self.log("Entrenamiento Q-Learning completado.")

    def entrenar_actor_critic(self, num_episodios):
        """Entrena el agente usando el algoritmo Actor-Critic."""
        self.log("Iniciando entrenamiento Actor-Critic...")
        recompensas_episodios = self.actor_critic.entrenar_agente(self.entorno, num_episodios=num_episodios)
        self.recompensas_totales = recompensas_episodios
        self.actualizar_grafico()
        self.log("Entrenamiento Actor-Critic completado.")


    def actualizar_grafico(self):
        """Actualiza el gráfico de recompensas."""
        self.ax.clear()
        self.ax.plot(self.recompensas_totales)
        self.ax.set_title("Curva de Aprendizaje")
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa total")
        self.canvas.draw()


def main():
    """Función principal para ejecutar la aplicación."""
    root = tk.Tk()
    aplicacion = Aplicacion(root)

    # Redirigir logging al widget de texto
    handler_text = TextHandler(aplicacion.txt_log)
    handler_text.setLevel(logging.INFO)
    logger.addHandler(handler_text)

    root.mainloop()

class TextHandler(logging.Handler):
    """
    Handler de logging personalizado para dirigir la salida a un widget de texto de Tkinter.
    """
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        log_entry = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, log_entry + '\n')
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

if __name__ == "__main__":
    main()
