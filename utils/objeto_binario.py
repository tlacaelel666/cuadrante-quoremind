import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import re
import numpy as np
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque  # Import deque for memory replay
# Configuración de logging mejorada
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logger's level

# Create a file handler and set level to debug
log_file_path = 'quantum_agent.log'
if os.path.exists(log_file_path):
    try:
        os.remove(log_file_path)
    except PermissionError as e:
        print(f"No se pudo eliminar el archivo de log existente {log_file_path}: {e}.  Continuando sin eliminar.")
    except Exception as e:
        print(f"Error inesperado al intentar eliminar {log_file_path}: {e}")

fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Show INFO and above in console

# Add formatter to handlers
fh.setFormatter(CustomFormatter())
ch.setFormatter(CustomFormatter())

# Add handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# Modelos de Agente
class ObjetoBinario:
    def __init__(self, nombre, num_categorias=5, bits_por_categoria=4):
        self.nombre = nombre
        self.num_categorias = num_categorias
        self.bits_por_categoria = bits_por_categoria
        self.categorias = ["0" * bits_por_categoria] * num_categorias

    def actualizar_categoria(self, indice, valor):
        if not (0 <= indice < self.num_categorias):
            raise ValueError(f"Índice de categoría inválido. Debe estar entre 0 y {self.num_categorias - 1}.")
        if not (0 <= int(valor) <= 2**self.bits_por_categoria - 1):
            raise ValueError(f"Valor fuera de rango. Debe estar entre 0 y {2**self.bits_por_categoria - 1}.")
        self.categorias[indice] = bin(int(valor))[2:].zfill(self.bits_por_categoria)

    def obtener_binario(self):
        return "".join(self.categorias)

    def obtener_categorias(self):
        return self.categorias

    def to_dict(self):
        return {
            "nombre": self.nombre,
            "num_categorias": self.num_categorias,  # Guardar num_categorias
            "bits_por_categoria": self.bits_por_categoria, # Guardar bits_por_categoria
            "categorias": self.categorias
        }

    @classmethod
    def from_dict(cls, data):
        # Usa los valores guardados para la inicialización
        obj = cls(data["nombre"], data["num_categorias"], data["bits_por_categoria"])
        obj.categorias = data["categorias"]
        return obj

class EntornoSimulado:
    def __init__(self, objetos, recompensa_por_movimiento=0.1, recompensa_por_ajuste=0.2):
        self.objetos = objetos
        self.estado_actual = 0
        self.recompensa_por_movimiento = recompensa_por_movimiento
        self.recompensa_por_ajuste = recompensa_por_ajuste
        self.num_acciones_disponibles = 4 + (objetos[0].num_categorias -1) if objetos else 4  # Añade acciones por categoría

    def obtener_estado(self):
        objeto_actual = self.objetos[self.estado_actual]
        estado_numerico = int(objeto_actual.obtener_binario(), 2)
        return estado_numerico

    def ejecutar_accion(self, accion):
        objeto_actual = self.objetos[self.estado_actual]
        num_categorias = objeto_actual.num_categorias
        
        if accion == 0:  # Mover a la derecha
            self.estado_actual = (self.estado_actual + 1) % len(self.objetos)
            recompensa = self.recompensa_por_movimiento
        elif accion == 1:  # Mover a la izquierda
            self.estado_actual = (self.estado_actual - 1) % len(self.objetos)
            recompensa = self.recompensa_por_movimiento
        elif accion == 2:  # Incrementar subcategoría 0 (ya existente)
            valor = int(objeto_actual.obtener_categorias()[0], 2)
            nuevo_valor = min(2**objeto_actual.bits_por_categoria - 1, valor + 1)
            objeto_actual.actualizar_categoria(0, str(nuevo_valor))
            recompensa = self.recompensa_por_ajuste
        elif accion == 3:  # Decrementar subcategoría 0 (ya existente)
            valor = int(objeto_actual.obtener_categorias()[0], 2)
            nuevo_valor = max(0, valor - 1)
            objeto_actual.actualizar_categoria(0, str(nuevo_valor))
            recompensa = self.recompensa_por_ajuste
        elif 4 <= accion < 4 + num_categorias :  # Acciones para incrementar/decrementar otras categorías
            indice_categoria = accion - 4
            valor = int(objeto_actual.obtener_categorias()[indice_categoria], 2)

            if (accion -2) % 2 == 0:
                nuevo_valor = min(2**objeto_actual.bits_por_categoria -1, valor + 1)
                recompensa = self.recompensa_por_ajuste
            
            else:
                nuevo_valor = max(0, valor - 1)
                recompensa = self.recompensa_por_ajuste
            
            objeto_actual.actualizar_categoria(indice_categoria, str(nuevo_valor))
            

        else:
            recompensa = -0.1  # Penalización por acción inválida
            logger.warning(f"Acción inválida: {accion}")

        siguiente_estado_numerico = self.obtener_estado()
        return siguiente_estado_numerico, recompensa, siguiente_estado_numerico


    def obtener_texto_estado(self):
        objeto_actual = self.objetos[self.estado_actual]
        categorias_str = ", ".join([f"Subcat{i+1}: {int(cat, 2)}" for i, cat in enumerate(objeto_actual.obtener_categorias())])
        return f"Objeto: {objeto_actual.nombre}. {categorias_str}"

    def reset(self):
        self.estado_actual = 0
        return self.obtener_estado()

# Modelos de Redes Neuronales
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], dropout_rate=0.2):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], dropout_rate=0.2):
        super(ActorCritic, self).__init__()
        # Actor network
        actor_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.ReLU())
            actor_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))  # Output is a single value
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# Agentes de Aprendizaje por Refuerzo
class QLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], learning_rate=0.001, gamma=0.99, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, batch_size=64, tau=0.001):
        self.q_network = QNetwork(state_dim, action_dim, hidden_layers)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_layers) # Target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights
        self.target_q_network.eval() # Set target network to evaluation mode
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=buffer_size)  # Replay buffer
        self.batch_size = batch_size
        self.tau = tau  # For soft updates of target network

    def seleccionar_accion(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()
    
    def almacenar_experiencia(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def aprender(self):
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples to learn from yet

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  # (batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) # (batch_size, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1)
        
        # Compute the target Q values
        with torch.no_grad():
            # max_q_next = self.q_network(next_states).max(dim=1)[0]
            max_q_next = self.target_q_network(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_q_next

        # Get current Q values
        current_q = self.q_network(states).gather(1, actions)  # (batch_size, 1)

        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update of target network
        self.soft_update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
    
    def soft_update_target_network(self):
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def guardar_modelo(self, ruta):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_q_network.state_dict(), # Save target network
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'memory': self.memory  # Save the replay buffer
        }, ruta)

    def cargar_modelo(self, ruta, hidden_layers):
      checkpoint = torch.load(ruta)
      self.q_network = QNetwork(checkpoint['state_dim'], checkpoint['action_dim'], hidden_layers)
      self.q_network.load_state_dict(checkpoint['model_state_dict'])
      
      # Load the target network
      self.target_q_network = QNetwork(checkpoint['state_dim'], checkpoint['action_dim'], hidden_layers)
      self.target_q_network.load_state_dict(checkpoint['target_model_state_dict'])
      self.target_q_network.eval()  # Ensure it's in eval mode
      
      self.optimizer = optim.Adam(self.q_network.parameters())
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.epsilon = checkpoint['epsilon']
      self.gamma = checkpoint['gamma']
      self.state_dim = checkpoint['state_dim']
      self.action_dim = checkpoint['action_dim']
      self.memory = checkpoint['memory']  # Load the replay buffer

class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 64], learning_rate=0.001, gamma=0.99):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_layers)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.action_dim = action_dim
        self.state_dim = state_dim

    def seleccionar_accion(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float32)
        policy, value = self.actor_critic(state_tensor)
        action_probs = F.softmax(policy, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Guardar log_prob y value para el entrenamiento
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        
        return action.item()
    
    def guardar_recompensa(self, reward):
        self.rewards.append(reward)
    
    def calcular_returns(self, next_value=0):
        returns = []
        R = next_value
        
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
            
        return returns
    
    def update(self, next_state=None):
        if len(self.rewards) == 0:
            return
            
        # Calcular valor del próximo estado si existe
        next_value = 0
        if next_state is not None:
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
            _, next_value = self.actor_critic(next_state_tensor)
            next_value = next_value.detach().item()
            
        # Calcular retornos
        returns = self.calcular_returns(next_value)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Convertir listas a tensores
        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values)
        
        # Calcular ventajas
        advantages = returns - values.detach()
        
        # Calcular pérdidas
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        
        # Pérdida total con balance entre actor y crítico
        loss = actor_loss + 0.5 * critic_loss
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Limpiar memoria
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return loss.item()
    
    def guardar_modelo(self, ruta):
        torch.save(self.actor_critic.state_dict(), ruta)
    
    def cargar_modelo(self, ruta):
        self.actor_critic.load_state_dict(torch.load(ruta))
        self.actor_critic.eval()
    
    def reset(self):
        self.log_probs = []
        self.values = []
        self.rewards = []

# Clase Categorical para A2CAgent
class Categorical:
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1).squeeze()

    def log_prob(self, action):
        return torch.log(self.probs.squeeze()[action])

# Clase principal de la aplicación
class QuantumAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Agent Simulator")
        self.root.geometry("1200x800")
        
        # Variables de control
        self.entorno = None
        self.agente = None
        self.agente_tipo = tk.StringVar(value="Q-Learning")
        self.entrenamiento_activo = False
        self.objetos = []
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.historial_epsilon = []
        self.ultimo_episodio = 0
        
        # Configurar estructura de la interfaz
        self.crear_estructura_ui()
        
        # Inicializar objetos por defecto
        self.crear_objetos_predeterminados()
        
        # Actualizar la información de estado
        self.actualizar_info_estado()

    def crear_estructura_ui(self):
        # Panel principal
        panel_principal = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panel_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo (controles)
        panel_izquierdo = ttk.Frame(panel_principal)
        panel_principal.add(panel_izquierdo, weight=30)
        
        # Panel derecho (visualización)
        panel_derecho = ttk.Frame(panel_principal)
        panel_principal.add(panel_derecho, weight=70)
        
        # Configuración de los paneles
        self.configurar_panel_izquierdo(panel_izquierdo)
        self.configurar_panel_derecho(panel_derecho)

    def configurar_panel_izquierdo(self, panel):
        # Frame para la información del entorno
        frame_entorno = ttk.LabelFrame(panel, text="Entorno")
        frame_entorno.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Información de estado actual
        ttk.Label(frame_entorno, text="Estado actual:").pack(anchor=tk.W, padx=5, pady=2)
        self.lbl_estado_actual = ttk.Label(frame_entorno, text="No iniciado")
        self.lbl_estado_actual.pack(anchor=tk.W, padx=5, pady=2)
        
        # Lista de objetos
        ttk.Label(frame_entorno, text="Objetos:").pack(anchor=tk.W, padx=5, pady=2)
        
        frame_objetos = ttk.Frame(frame_entorno)
        frame_objetos.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(frame_objetos)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree_objetos = ttk.Treeview(frame_objetos, columns=("Nombre", "Categorías"), show="headings")
        self.tree_objetos.heading("Nombre", text="Nombre")
        self.tree_objetos.heading("Categorías", text="Categorías")
        self.tree_objetos.column("Nombre", width=100)
        self.tree_objetos.column("Categorías", width=150)
        self.tree_objetos.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.tree_objetos.yview)
        self.tree_objetos.config(yscrollcommand=scrollbar.set)
        
        # Botones para gestionar objetos
        frame_botones_objetos = ttk.Frame(frame_entorno)
        frame_botones_objetos.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frame_botones_objetos, text="Añadir", command=self.mostrar_dialogo_nuevo_objeto).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_objetos, text="Editar", command=self.editar_objeto_seleccionado).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_objetos, text="Eliminar", command=self.eliminar_objeto_seleccionado).pack(side=tk.LEFT, padx=2)
        
        # Frame para configuración del agente
        frame_agente = ttk.LabelFrame(panel, text="Configuración del Agente")
        frame_agente.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tipo de agente
        ttk.Label(frame_agente, text="Tipo de Agente:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(frame_agente, text="Q-Learning", variable=self.agente_tipo, value="Q-Learning").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(frame_agente, text="Actor-Crítico (A2C)", variable=self.agente_tipo, value="A2C").pack(anchor=tk.W, padx=20, pady=2)
        
        # Parámetros de entrenamiento
        ttk.Label(frame_agente, text="Hiperparámetros:").pack(anchor=tk.W, padx=5, pady=2)
        
        frame_params = ttk.Frame(frame_agente)
        frame_params.pack(fill=tk.X, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(frame_params, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_lr = ttk.Entry(frame_params, width=10)
        self.entry_lr.insert(0, "0.001")
        self.entry_lr.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Gamma
        ttk.Label(frame_params, text="Gamma:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_gamma = ttk.Entry(frame_params, width=10)
        self.entry_gamma.insert(0, "0.99")
        self.entry_gamma.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Epsilon (para Q-Learning)
        ttk.Label(frame_params, text="Epsilon inicial:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_epsilon = ttk.Entry(frame_params, width=10)
        self.entry_epsilon.insert(0, "0.99")
        self.entry_epsilon.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Epsilon decay
        ttk.Label(frame_params, text="Epsilon decay:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_epsilon_decay = ttk.Entry(frame_params, width=10)
        self.entry_epsilon_decay.insert(0, "0.995")
        self.entry_epsilon_decay.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Arquitectura de red
        ttk.Label(frame_agente, text="Arquitectura de Red:").pack(anchor=tk.W, padx=5, pady=2)
        
        frame_red = ttk.Frame(frame_agente)
        frame_red.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(frame_red, text="Capas ocultas (separadas por comas):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.entry_hidden_layers = ttk.Entry(frame_red, width=15)
        self.entry_hidden_layers.insert(0, "128, 64")
        self.entry_hidden_layers.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Botones de entrenamiento
        frame_botones_entrenamiento = ttk.Frame(frame_agente)
        frame_botones_entrenamiento.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_iniciar = ttk.Button(frame_botones_entrenamiento, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento)
        self.btn_iniciar.pack(side=tk.LEFT, padx=2)
        
        self.btn_detener = ttk.Button(frame_botones_entrenamiento, text="Detener", command=self.detener_entrenamiento, state=tk.DISABLED)
        self.btn_detener.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(frame_botones_entrenamiento, text="Reiniciar", command=self.reiniciar_entorno).pack(side=tk.LEFT, padx=2)
        
        # Frame para acciones manuales
        frame_acciones = ttk.LabelFrame(panel, text="Acciones Manuales")
        frame_acciones.pack(fill=tk.X, padx=5, pady=5)
        
        # Botones de acciones
        frame_botones_acciones = ttk.Frame(frame_acciones)
        frame_botones_acciones.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frame_botones_acciones, text="◀", command=lambda: self.ejecutar_accion_manual(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_acciones, text="▶", command=lambda: self.ejecutar_accion_manual(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_acciones, text="▲", command=lambda: self.ejecutar_accion_manual(2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_botones_acciones, text="▼", command=lambda: self.ejecutar_accion_manual(3)).pack(side=tk.LEFT, padx=2)
        
        # Botones para guardar/cargar
        frame_guardar_cargar = ttk.Frame(panel)
        frame_guardar_cargar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frame_guardar_cargar, text="Guardar Modelo", command=self.guardar_modelo).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_guardar_cargar, text="Cargar Modelo", command=self.cargar_modelo).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_guardar_cargar, text="Exportar Entorno", command=self.exportar_entorno).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_guardar_cargar, text="Importar Entorno", command=self.importar_entorno).pack(side=tk.LEFT, padx=2)

    def configurar_panel_derecho(self, panel):
        # Notebook para pestañas
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de visualización de entrenamiento
        tab_entrenamiento = ttk.Frame(notebook)
        notebook.add(tab_entrenamiento, text="Entrenamiento")
        
        # Frame para gráficos
        frame_graficos = ttk.Frame(tab_entrenamiento)
        frame_graficos.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear figura para gráficos
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        self.fig.tight_layout(pad=3.0)
        
        # Gráfico de recompensas
        self.ax1.set_title("Recompensa por Episodio")
        self.ax1.set_ylabel("Recompensa")
        self.line_reward, = self.ax1.plot([], [], 'b-')
        
        # Gráfico de pérdidas
        self.ax2.set_title("Pérdida por Episodio")
        self.ax2.set_ylabel("Pérdida")
        self.line_loss, = self.ax2.plot([], [], 'r-')
        
        # Gráfico de epsilon (para Q-Learning)
        self.ax3.set_title("Epsilon por Episodio")
        self.ax3.set_xlabel("Episodio")
        self.ax3.set_ylabel("Epsilon")
        self.line_epsilon, = self.ax3.plot([], [], 'g-')
        
        # Crear canvas para mostrar los gráficos
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficos)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de log
        tab_log = ttk.Frame(notebook)
        notebook.add(tab_log, text="Log")
        
        # Área de texto para mostrar logs
        self.txt_log = scrolledtext.ScrolledText(tab_log)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar handler para mostrar logs en UI
        self.log_handler = LogTextHandler(self.txt_log)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(CustomFormatter())
        logger.addHandler(self.log_handler)

    def crear_objetos_predeterminados(self):
        # Crear algunos objetos por defecto
        objeto1 = ObjetoBinario("Objeto 1", num_categorias=3, bits_por_categoria=4)
        objeto2 = ObjetoBinario("Objeto 2", num_categorias=3, bits_por_categoria=4)
        objeto3 = ObjetoBinario("Objeto 3", num_categorias=3, bits_por_categoria=4)
        
        self.objetos = [objeto1, objeto2, objeto3]
        
        # Actualizar la vista de árbol
        self.actualizar_vista_objetos()
        
        # Crear entorno con estos objetos
        self.entorno = EntornoSimulado(self.objetos)

    def actualizar_vista_objetos(self):
        # Limpiar vista actual
        for item in self.tree_objetos.get_children():
            self.tree_objetos.delete(item)
        
        # Rellenar con los objetos actuales
        for objeto in self.objetos:
            categorias_str = ", ".join([str(int(cat, 2)) for cat in objeto.obtener_categorias()])
            self.tree_objetos.insert("", tk.END, values=(objeto.nombre, categorias_str))

    def mostrar_dialogo_nuevo_objeto(self):
        # Crear diálogo para nuevo objeto
        dialogo = tk.Toplevel(self.root)
        dialogo.title("Nuevo Objeto")
        dialogo.geometry("300x250")
        dialogo.transient(self.root)
        dialogo.grab_set()
        
        # Campos del formulario
        ttk.Label(dialogo, text="Nombre:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry_nombre = ttk.Entry(dialogo, width=20)
        entry_nombre.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(dialogo, text="Número de categorías:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        entry_num_cat = ttk.Entry(dialogo, width=5)
        entry_num_cat.insert(0, "3")
        entry_num_cat.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(dialogo, text="Bits por categoría:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        entry_bits = ttk.Entry(dialogo, width=5)
        entry_bits.insert(0, "4")
        entry_bits.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Botones
        frame_botones = ttk.Frame(dialogo)
        frame_botones.grid(row=3, column=0, columnspan=2, pady=10)
        
        def guardar_objeto():
            try:
                nombre = entry_nombre.get().strip()
                if not nombre:
                    messagebox.showerror("Error", "El nombre no puede estar vacío")
                    return
                
                num_cat = int(entry_num_cat.get())
                bits = int(entry_bits.get())
                
                if num_cat <= 0 or bits <= 0:
                    messagebox.showerror("Error", "Los valores deben ser positivos")
                    return
                
                # Crear nuevo objeto
                nuevo_objeto = ObjetoBinario(nombre, num_cat, bits)
                self.objetos.append(nuevo_objeto)
                
                # Actualizar vista
                self.actualizar_vista_objetos()
                
                # Recrear entorno con los nuevos objetos
                self.entorno = EntornoSimulado(self.objetos)
                
                # Actualizar estado
                self.actualizar_info_estado()
                
                # Cerrar diálogo
                dialogo.destroy()
                
            except ValueError as e:
                messagebox.showerror("Error", f"Valor inválido: {e}")
        
        ttk.Button(frame_botones, text="Guardar", command=guardar_objeto).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Cancelar", command=dialogo.destroy).pack(side=tk.LEFT, padx=5)

    def editar_objeto_seleccionado(self):
        # Obtener objeto seleccionado
        seleccion = self.tree_objetos.selection()
        if not seleccion:
            messagebox.showinfo("Información", "Seleccione un objeto para editar")
            return
        
        indice = self.tree_objetos.index(seleccion[0])
        objeto = self.objetos[indice]
        
        # Crear diálogo para editar
        dialogo = tk.Toplevel(self.root)
        dialogo.title(f"Editar {objeto.nombre}")
        dialogo.geometry("400x400")
        dialogo.transient(self.root)
        dialogo.grab_set()
        
        # Campos básicos
        ttk.Label(dialogo, text="Nombre:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        entry_nombre = ttk.Entry(dialogo, width=20)
        entry_nombre.insert(0, objeto.nombre)
        entry_nombre.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # No permitimos cambiar número de categorías o bits (por simplicidad)
        ttk.Label(dialogo, text=f"Categorías: {objeto.num_categorias}").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(dialogo, text=f"Bits por categoría: {objeto.bits_por_categoria}").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Frame para valores de categorías
        frame_categorias = ttk.LabelFrame(dialogo, text="Valores de Categorías")
        frame_categorias.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Campos para cada categoría
        entries_categoria = []
        for i, categoria in enumerate(objeto.obtener_categorias()):
            ttk.Label(frame_categorias, text=f"Categoría {i+1}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry_cat = ttk.Entry(frame_categorias, width=10)
            entry_cat.insert(0, str(int(categoria, 2)))
            entry_cat.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            entries_categoria.append(entry_cat)
        
        # Botones
        frame_botones = ttk.Frame(dialogo)
        frame_botones.grid(row=4, column=0, columnspan=2, pady=10)
        
        def guardar_cambios():
            try:
                # Actualizar nombre
                objeto.nombre = entry_nombre.get().strip()
                
                # Actualizar valores de categorías
                for i, entry in enumerate(entries_categoria):
                    valor = int(entry.get())
                    objeto.actualizar_categoria(i, str(valor))
                
                # Actualizar vista
                self.actualizar_vista_objetos()
                
                # Actualizar estado
                self.actualizar_info_estado()
                
                # Cerrar diálogo
                dialogo.destroy()
                
            except ValueError as e:
                messagebox.showerror("Error", f"Valor inválido: {e}")
        
        ttk.Button(frame_botones, text="Guardar", command=guardar_cambios).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Cancelar", command=dialogo.destroy).pack(side=tk.LEFT, padx=5)

    def eliminar_objeto_seleccionado(self):
        # Obtener objeto seleccionado
        seleccion = self.tree_objetos.selection()
        if not seleccion:
            messagebox.showinfo("Información", "Seleccione un objeto para eliminar")
            return
        
        indice = self.tree_objetos.index(seleccion[0])
        
        # Confirmar eliminación
        if messagebox.askyesno("Confirmar", f"¿Seguro que desea eliminar {self.objetos[indice].nombre}?"):
            # Eliminar objeto
            del self.objetos[indice]
            
            # Si no quedan objetos, crear uno por defecto
            if not self.objetos:
                self.objetos.append(ObjetoBinario("Objeto Default", 3, 4))
            
            # Actualizar vista
            self.actualizar_vista_objetos()
            
            # Recrear entorno
            self.entorno = EntornoSimulado(self.objetos)
            
            # Actualizar estado
            self.actualizar_info_estado()

    def actualizar_info_estado(self):
        if self.entorno:
            self.lbl_estado_actual.config(text=self.entorno.obtener_texto_estado())
            logger.info(f"Estado actual: {self.entorno.obtener_texto_estado()}")
        else:
            self.lbl_estado_actual.config(text="No iniciado")

    def iniciar_entrenamiento(self):
        if not self.entorno:
            messagebox.showerror("Error", "No hay un entorno configurado")
            return
        
        if self.entrenamiento_activo:
            return
        
        # Deshabilitar botón de inicio y habilitar botón de detener
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)
        
        # Configuración de hiperparámetros
        try:
            learning_rate = float(self.entry_lr.get())
            gamma = float(self.entry_gamma.get())
            epsilon = float(self.entry_epsilon.get())
            epsilon_decay = float(self.entry_epsilon_decay.get())
            
            # Arquitectura de red
            hidden_layers_str = self.entry_hidden_layers.get()
            hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(",")]
        except ValueError as e:
            messagebox.showerror("Error", f"Valor inválido en los parámetros: {e}")
            sel
