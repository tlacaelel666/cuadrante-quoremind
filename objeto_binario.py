
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import re

class ObjetoBinario:
    def __init__(self, nombre):
        self.nombre = nombre
        self.categorias = ["0000"] * 5  # Inicializamos 5 subcategorías vacías

    def actualizar_categoria(self, indice, valor):
        """Actualiza el valor de una subcategoría."""
        if 0 <= indice < 5 and 0 <= int(valor) <= 10:
            self.categorias[indice] = bin(int(valor))[2:].zfill(4) # Convertir a binario de 4 bits
        else:
            raise ValueError("Índice de categoría inválido o valor fuera de rango (0-10).")
    
    def obtener_binario(self):
        """Devuelve la representación binaria combinada."""
        return "".join(self.categorias)

    def obtener_categorias(self):
        """Devuelve la lista de categorias."""
        return self.categorias

# Definir la red neuronal
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EntornoSimulado:
  """Entorno simulado simple."""
  def __init__(self, objetos):
    self.objetos = objetos
    self.estado_actual = 0

  def obtener_estado(self):
    """Devuelve el estado actual (indice del objeto)."""
    return self.estado_actual

  def ejecutar_accion(self, accion):
    """Ejecuta una acción y devuelve el siguiente estado, recompensa y estado actual."""
    if accion == 0: # Accion mover a la derecha
      self.estado_actual = (self.estado_actual + 1) % len(self.objetos)
      recompensa = 1
    elif accion == 1: # Accion mover a la izquierda
      self.estado_actual = (self.estado_actual - 1) % len(self.objetos)
      recompensa = 1
    elif accion == 2: # Accion incrementar subcategoria 1 del objeto actual
      objeto_actual = self.objetos[self.estado_actual]
      valor = int(objeto_actual.obtener_categorias()[0],2)
      nuevo_valor = min(10, valor + 1) # Aumentar hasta 10
      try:
        objeto_actual.actualizar_categoria(0, str(nuevo_valor))
        recompensa = 2
      except:
         recompensa = -1
    elif accion == 3: # Accion decrementar subcategoria 1 del objeto actual
      objeto_actual = self.objetos[self.estado_actual]
      valor = int(objeto_actual.obtener_categorias()[0],2)
      nuevo_valor = max(0, valor - 1) # Disminuir hasta 0
      try:
         objeto_actual.actualizar_categoria(0, str(nuevo_valor))
         recompensa = 2
      except:
        recompensa = -1
    else:
      recompensa = -1 # Accion invalida
    
    return self.estado_actual, recompensa, self.obtener_estado()
  
  def obtener_texto_estado(self):
     return f"Objeto actual: {self.objetos[self.estado_actual].nombre}. Valor subcategoria 1: {int(self.objetos[self.estado_actual].obtener_categorias()[0],2)}"

class Aplicacion:
  def __init__(self, root):
        self.root = root
        self.root.title("Gestor, Agente RL y Entorno Simulado")

        self.objetos = [ObjetoBinario(f"Objeto {i+1}") for i in range(3)]
        self.entorno = EntornoSimulado(self.objetos)
        self.state_dim = 1  # Solo el indice del objeto actual
        self.action_dim = 4 # Mover izq, der, inc, dec subcat 1
        self.hidden_dim = 128
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.epsilon = 0.99 # Epsilon inicial
        self.epsilon_decay = 0.995
        self.gamma = 0.99 # Factor de descuento

        self.crear_interfaz()

  def crear_interfaz(self):
    # Etiqueta de título
    ttk.Label(self.root, text="Agente RL en Entorno de Objetos Binarios", font=("Arial", 16)).pack(pady=10)

    # Frame para entrada de texto
    frame_entrada = ttk.Frame(self.root)
    frame_entrada.pack(pady=10)

    ttk.Label(frame_entrada, text="Comando:").pack(side="left", padx=5)
    self.entry_comando = ttk.Entry(frame_entrada, width=30)
    self.entry_comando.pack(side="left", padx=5)

    btn_enviar = ttk.Button(frame_entrada, text="Enviar", command=self.procesar_comando)
    btn_enviar.pack(side="left", padx=5)

    # Frame para la salida
    frame_salida = ttk.Frame(self.root)
    frame_salida.pack(pady=10)

    ttk.Label(frame_salida, text="Retroalimentación:").pack()
    self.text_retroalimentacion = scrolledtext.ScrolledText(frame_salida, height=10, width=60)
    self.text_retroalimentacion.pack()

    #Boton para entrenar la red
    btn_entrenar = ttk.Button(self.root, text="Entrenar", command=self.entrenar_agente)
    btn_entrenar.pack(pady=10)


  def procesar_comando(self):
      """Procesa el comando del usuario usando PLN básico."""
      texto_comando = self.entry_comando.get().lower()
      accion = self.interpretar_comando(texto_comando)
      
      estado_previo = self.entorno.obtener_estado()
      estado_nuevo, recompensa, _ = self.entorno.ejecutar_accion(accion)
      texto_estado = self.entorno.obtener_texto_estado()
      self.text_retroalimentacion.insert(tk.END, f"Comando: {texto_comando} -> Acción: {accion}. {texto_estado}. Recompensa: {recompensa}\n")
      self.text_retroalimentacion.see(tk.END)  # Autoscroll
      self.aprender(estado_previo, accion, recompensa, estado_nuevo)

  def interpretar_comando(self, texto):
    """Interpreta comandos usando reglas básicas."""
    if re.search(r'\b(izquierda|atras)\b', texto):
      return 1 # mover izq
    elif re.search(r'\b(derecha|siguiente)\b', texto):
      return 0 # mover der
    elif re.search(r'\b(aumenta|sube|incrementa)\b', texto):
      return 2 # inc subcat
    elif re.search(r'\b(disminuye|baja|reduce)\b', texto):
      return 3 # dec subcat
    else:
      return random.choice([0,1,2,3]) # Accion aleatoria si no se reconoce

  def aprender(self, state, action, reward, next_state):
    """Realiza el aprendizaje usando Q-Learning."""
    self.q_network.train() # Modo entrenamiento
    state_tensor = torch.tensor([state], dtype=torch.float32)
    next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
    
    with torch.no_grad():
        max_q_next = self.q_network(next_state_tensor).max(dim=1)[0]
        target_q = reward + self.gamma * max_q_next
    
    q_values = self.q_network(state_tensor)
    loss = F.mse_loss(q_values[0, action], target_q)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  
  def seleccionar_accion(self, state):
    """Selecciona una acción usando epsilon-greedy."""
    if random.random() < self.epsilon:
        return random.randint(0, self.action_dim-1)
    else:
       self.q_network.eval() # Modo evaluacion
       with torch.no_grad():
          state_tensor = torch.tensor([state], dtype=torch.float32)
          q_values = self.q_network(state_tensor)
          return torch.argmax(q_values).item()
  
  def entrenar_agente(self):
    """Entrena al agente usando un loop de interacciones."""
    num_epocas = 500
    for epoca in range(num_epocas):
      estado_actual = self.entorno.obtener_estado()
      accion = self.seleccionar_accion(estado_actual)
      
      nuevo_estado, recompensa, _ = self.entorno.ejecutar_accion(accion)
      self.aprender(estado_actual, accion, recompensa, nuevo_estado)
      
      if self.epsilon > 0.1:
        self.epsilon *= self.epsilon_decay
      
      if (epoca+1) % 100 == 0:
         print(f"Epoca: {epoca+1}, Epsilon: {self.epsilon:.2f}")
    
    print("Entrenamiento completado.")

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()


### implementacion con a2c
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import re
import numpy as np

class ObjetoBinario:
    def __init__(self, nombre):
        self.nombre = nombre
        self.categorias = ["0000"] * 5  # Inicializamos 5 subcategorías vacías

    def actualizar_categoria(self, indice, valor):
        """Actualiza el valor de una subcategoría."""
        if 0 <= indice < 5 and 0 <= int(valor) <= 10:
            self.categorias[indice] = bin(int(valor))[2:].zfill(4) # Convertir a binario de 4 bits
        else:
            raise ValueError("Índice de categoría inválido o valor fuera de rango (0-10).")
    
    def obtener_binario(self):
        """Devuelve la representación binaria combinada."""
        return "".join(self.categorias)

    def obtener_categorias(self):
        """Devuelve la lista de categorias."""
        return self.categorias

# Definir las redes neuronales (Actor y Critic)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Actor network (policy network)
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)
        # Critic network (value network)
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)  # Single value output
    
    def forward(self, x):
        # Actor (policy network)
        actor_x = F.relu(self.actor_fc1(x))
        action_probs = F.softmax(self.actor_fc2(actor_x), dim=-1)

        # Critic (value network)
        critic_x = F.relu(self.critic_fc1(x))
        state_value = self.critic_fc2(critic_x)

        return action_probs, state_value

class EntornoSimulado:
  """Entorno simulado simple."""
  def __init__(self, objetos):
    self.objetos = objetos
    self.estado_actual = 0

  def obtener_estado(self):
    """Devuelve el estado actual (indice del objeto)."""
    return self.estado_actual

  def ejecutar_accion(self, accion):
    """Ejecuta una acción y devuelve el siguiente estado, recompensa y estado actual."""
    if accion == 0: # Accion mover a la derecha
      self.estado_actual = (self.estado_actual + 1) % len(self.objetos)
      recompensa = 1
    elif accion == 1: # Accion mover a la izquierda
      self.estado_actual = (self.estado_actual - 1) % len(self.objetos)
      recompensa = 1
    elif accion == 2: # Accion incrementar subcategoria 1 del objeto actual
      objeto_actual = self.objetos[self.estado_actual]
      valor = int(objeto_actual.obtener_categorias()[0],2)
      nuevo_valor = min(10, valor + 1) # Aumentar hasta 10
      try:
        objeto_actual.actualizar_categoria(0, str(nuevo_valor))
        recompensa = 2
      except:
         recompensa = -1
    elif accion == 3: # Accion decrementar subcategoria 1 del objeto actual
      objeto_actual = self.objetos[self.estado_actual]
      valor = int(objeto_actual.obtener_categorias()[0],2)
      nuevo_valor = max(0, valor - 1) # Disminuir hasta 0
      try:
         objeto_actual.actualizar_categoria(0, str(nuevo_valor))
         recompensa = 2
      except:
        recompensa = -1
    else:
      recompensa = -1 # Accion invalida
    
    return self.estado_actual, recompensa, self.obtener_estado()
  
  def obtener_texto_estado(self):
     return f"Objeto actual: {self.objetos[self.estado_actual].nombre}. Valor subcategoria 1: {int(self.objetos[self.estado_actual].obtener_categorias()[0],2)}"

class Aplicacion:
  def __init__(self, root):
        self.root = root
        self.root.title("Gestor, Agente A2C y Entorno Simulado")

        self.objetos = [ObjetoBinario(f"Objeto {i+1}") for i in range(3)]
        self.entorno = EntornoSimulado(self.objetos)
        self.state_dim = 1  # Solo el indice del objeto actual
        self.action_dim = 4 # Mover izq, der, inc, dec subcat 1
        self.hidden_dim = 128
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        self.gamma = 0.99
        self.log_probs = []
        self.values = []
        self.rewards = []

        self.crear_interfaz()

  def crear_interfaz(self):
    # Etiqueta de título
    ttk.Label(self.root, text="Agente A2C en Entorno de Objetos Binarios", font=("Arial", 16)).pack(pady=10)

    # Frame para entrada de texto
    frame_entrada = ttk.Frame(self.root)
    frame_entrada.pack(pady=10)

    ttk.Label(frame_entrada, text="Comando:").pack(side="left", padx=5)
    self.entry_comando = ttk.Entry(frame_entrada, width=30)
    self.entry_comando.pack(side="left", padx=5)

    btn_enviar = ttk.Button(frame_entrada, text="Enviar", command=self.procesar_comando)
    btn_enviar.pack(side="left", padx=5)

    # Frame para la salida
    frame_salida = ttk.Frame(self.root)
    frame_salida.pack(pady=10)

    ttk.Label(frame_salida, text="Retroalimentación:").pack()
    self.text_retroalimentacion = scrolledtext.ScrolledText(frame_salida, height=10, width=60)
    self.text_retroalimentacion.pack()

    #Boton para entrenar la red
    btn_entrenar = ttk.Button(self.root, text="Entrenar", command=self.entrenar_agente)
    btn_entrenar.pack(pady=10)


  def procesar_comando(self):
      """Procesa el comando del usuario usando PLN básico."""
      texto_comando = self.entry_comando.get().lower()
      accion = self.interpretar_comando(texto_comando)
      
      estado_previo = self.entorno.obtener_estado()
      estado_nuevo, recompensa, _ = self.entorno.ejecutar_accion(accion)
      texto_estado = self.entorno.obtener_texto_estado()
      self.text_retroalimentacion.insert(tk.END, f"Comando: {texto_comando} -> Acción: {accion}. {texto_estado}. Recompensa: {recompensa}\n")
      self.text_retroalimentacion.see(tk.END)  # Autoscroll
      self.almacenar_experiencia(estado_previo, accion, recompensa)

  def interpretar_comando(self, texto):
    """Interpreta comandos usando reglas básicas."""
    if re.search(r'\b(izquierda|atras)\b', texto):
      return 1 # mover izq
    elif re.search(r'\b(derecha|siguiente)\b', texto):
      return 0 # mover der
    elif re.search(r'\b(aumenta|sube|incrementa)\b', texto):
      return 2 # inc subcat
    elif re.search(r'\b(disminuye|baja|reduce)\b', texto):
      return 3 # dec subcat
    else:
      return random.choice([0,1,2,3]) # Accion aleatoria si no se reconoce

  def almacenar_experiencia(self, state, action, reward):
    """Almacena la experiencia para el entrenamiento."""
    state_tensor = torch.tensor([state], dtype=torch.float32)
    action_probs, state_value = self.actor_critic(state_tensor)
    
    action_prob = action_probs[0, action]
    log_prob = torch.log(action_prob)
    
    self.log_probs.append(log_prob)
    self.values.append(state_value)
    self.rewards.append(reward)

  def calcular_retorno(self, rewards, gamma):
    """Calcula la recompensa acumulada."""
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
      cumulative_reward = reward + gamma * cumulative_reward
      discounted_rewards.insert(0, cumulative_reward)
    return torch.tensor(discounted_rewards, dtype=torch.float32)
  
  def actualizar_red(self, discounted_rewards, values, log_probs, gamma):
    """Actualiza las redes del Actor y el Critic."""
    
    values = torch.cat(values).squeeze() # Convert to tensor
    returns = discounted_rewards # Torch.tensor
    advantages = returns - values # Advantage
    
    log_probs = torch.cat(log_probs).squeeze()
    actor_loss = - (log_probs * advantages).mean()
    critic_loss = F.mse_loss(values, returns)
    
    total_loss = actor_loss + critic_loss
    
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()

    return actor_loss, critic_loss
  
  def seleccionar_accion(self, state):
    """Selecciona una acción usando la política del Actor."""
    self.actor_critic.eval() # Modo evaluacion
    with torch.no_grad():
      state_tensor = torch.tensor([state], dtype=torch.float32)
      action_probs, _ = self.actor_critic(state_tensor)
      action = torch.multinomial(action_probs, 1).item()
      return action
  
  def entrenar_agente(self):
    """Entrena al agente usando el algoritmo A2C."""
    num_epocas = 500
    for epoca in range(num_epocas):
      self.log_probs.clear()
      self.values.clear()
      self.rewards.clear()
      estado_actual = self.entorno.obtener_estado()
      accion = self.seleccionar_accion(estado_actual)
      
      nuevo_estado, recompensa, _ = self.entorno.ejecutar_accion(accion)
      
      texto_estado = self.entorno.obtener_texto_estado()
      self.text_retroalimentacion.insert(tk.END, f"Acción: {accion}. {texto_estado}. Recompensa: {recompensa}\n")
      self.text_retroalimentacion.see(tk.END)
      self.almacenar_experiencia(estado_actual, accion, recompensa)

      #Actualizar redes en cada iteracion
      discounted_rewards = self.calcular_retorno(self.rewards, self.gamma)
      actor_loss, critic_loss = self.actualizar_red(discounted_rewards, self.values, self.log_probs, self.gamma)
      if (epoca+1) % 100 == 0:
          print(f"Epoca: {epoca+1}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    
    print("Entrenamiento completado.")

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()
