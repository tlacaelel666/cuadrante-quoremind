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
