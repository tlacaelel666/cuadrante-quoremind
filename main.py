#Crear entorno virtual 
#!pip install virtualenv 
#virtualenv env
#source env/bin/activate 
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
import circuito_principal
import modelo_hibrido

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

if __name__ == "__main__":
    main()
inbox_lr.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        
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
