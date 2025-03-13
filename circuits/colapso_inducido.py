from typing import Tuple, Dict, Any, List, Optional
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulacion.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Asegurarse de que los módulos personalizados estén en el path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from logic import BayesLogic, shannon_entropy, calculate_cosines
    logger.info("Módulos de lógica importados correctamente")
except ImportError as e:
    logger.error(f"Error al importar módulos de logic.py: {e}")
    logger.info("Implementando funciones de respaldo en caso de que logic.py no esté disponible")
    
    # Implementaciones de respaldo en caso de que logic.py no esté disponible
    class BayesLogic:
        """Clase de respaldo para BayesLogic en caso de que logic.py no esté disponible."""
        def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, 
                                                     prn_influence: float, action: int) -> Dict[str, Any]:
            """Versión simplificada del cálculo de probabilidades."""
            prob_action_0 = 0.5 - (0.1 * entropy) + (0.1 * coherence) - (0.05 * prn_influence)
            prob_action_1 = 1 - prob_action_0
            
            # Asegurar que las probabilidades estén en el rango [0, 1]
            prob_action_0 = max(0, min(1, prob_action_0))
            prob_action_1 = max(0, min(1, prob_action_1))
            
            # Normalizar las probabilidades
            total = prob_action_0 + prob_action_1
            if total > 0:
                prob_action_0 /= total
                prob_action_1 /= total
            else:
                prob_action_0 = prob_action_1 = 0.5
            
            # Seleccionar acción basada en probabilidades
            if np.random.random() < prob_action_0:
                action_to_take = 0
            else:
                action_to_take = 1
                
            return {
                "prob_action_0": prob_action_0,
                "prob_action_1": prob_action_1,
                "action_to_take": action_to_take
            }
    
    def shannon_entropy(signal: np.ndarray) -> float:
        """Versión de respaldo para el cálculo de la entropía de Shannon."""
        signal = np.abs(signal)
        total = np.sum(signal)
        if total == 0:
            return 0
        
        # Normalizar para obtener una distribución de probabilidad
        prob = signal / total
        
        # Eliminar valores donde prob es 0 para evitar log(0)
        prob = prob[prob > 0]
        
        # Calcular entropía
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def calculate_cosines(entropy: float, env_value: float) -> Tuple[float, float, float]:
        """Versión de respaldo para calcular cosenos directores."""
        # Valores simplificados basados en la entropía y el valor ambiental
        cos_x = 0.5 * entropy + 0.2 * env_value
        cos_y = 0.3 * entropy + 0.4 * env_value
        cos_z = 0.2 * entropy + 0.6 * env_value
        
        # Normalizar para que la suma de los cuadrados sea 1
        magnitude = np.sqrt(cos_x**2 + cos_y**2 + cos_z**2)
        if magnitude > 0:
            cos_x /= magnitude
            cos_y /= magnitude
            cos_z /= magnitude
        
        return cos_x, cos_y, cos_z


# --- Clase PRN (Probabilistic Reference Noise) ---
@dataclass
class PRN:
    """
    Clase para simular el Ruido Probabilístico de Referencia (PRN).
    
    Attributes:
        influence (float): Nivel de influencia del ruido, en el rango [0, 1].
        seed (Optional[int]): Semilla para el generador de números aleatorios.
    """
    influence: float = 0.5
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validación después de inicialización."""
        if not 0 <= self.influence <= 1:
            logger.warning(f"Valor de influencia fuera de rango: {self.influence}, ajustando al rango [0, 1]")
            self.influence = max(0, min(1, self.influence))
        
        if self.seed is not None:
            np.random.seed(self.seed)
            logger.info(f"PRN inicializado con semilla: {self.seed}")
    
    def get_noise(self) -> float:
        """Genera un valor de ruido aleatorio."""
        return np.random.random() * self.influence
    
    def update_influence(self, new_influence: float) -> None:
        """Actualiza el nivel de influencia del ruido."""
        if not 0 <= new_influence <= 1:
            logger.warning(f"Nuevo valor de influencia fuera de rango: {new_influence}, ajustando al rango [0, 1]")
            new_influence = max(0, min(1, new_influence))
        
        self.influence = new_influence
        logger.debug(f"Influencia del PRN actualizada a: {self.influence}")


# --- Funciones auxiliares para la visualización de la red neuronal (mejoradas) ---
@dataclass
class NeuralNode:
    """
    Representación de un nodo de la red neuronal.
    
    Attributes:
        active (bool): Estado de activación del nodo.
        activation_value (float): Valor de activación (puede ser útil para visualizaciones más detalladas).
        connections (List[Tuple[int, int]]): Lista de conexiones a otros nodos (capa, índice).
    """
    active: bool = False
    activation_value: float = 0.0
    connections: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Inicialización post-inicialización."""
        if self.connections is None:
            self.connections = []
    
    def activate(self, value: float = 1.0) -> None:
        """Activa el nodo con un valor específico."""
        self.active = True
        self.activation_value = value
    
    def deactivate(self) -> None:
        """Desactiva el nodo."""
        self.active = False
        self.activation_value = 0.0
    
    def is_active(self) -> bool:
        """Verifica si el nodo está activo."""
        return self.active
    
    def add_connection(self, layer_idx: int, node_idx: int) -> None:
        """Añade una conexión a otro nodo."""
        self.connections.append((layer_idx, node_idx))


def initialize_node() -> NeuralNode:
    """Inicializa un nodo de la red neuronal."""
    return NeuralNode()


def is_active(node: NeuralNode) -> bool:
    """Verifica si un nodo está activo."""
    return node.is_active()


# --- Clase TimeSeries Mejorada ---
class TimeSeries:
    """
    Clase para representar y manipular series temporales sinusoidales.
    
    Attributes:
        amplitud (float): Amplitud de la onda sinusoidal.
        frecuencia (float): Frecuencia de la onda sinusoidal.
        fase (float): Fase de la onda sinusoidal en radianes.
        nombre (str): Nombre descriptivo de la serie (útil para visualizaciones).
    """
    def __init__(self, amplitud: float, frecuencia: float, fase: float, nombre: str = ""):
        """Inicializa la serie temporal sinusoidal."""
        self.amplitud = amplitud
        self.frecuencia = frecuencia
        self.fase = fase
        self.nombre = nombre
        logger.debug(f"Serie temporal '{nombre}' inicializada: A={amplitud}, f={frecuencia}, φ={fase}")
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evalúa la onda sinusoidal en los puntos x."""
        return self.amplitud * np.sin(2 * np.pi * self.frecuencia * x + self.fase)
    
    def get_phase(self) -> float:
        """Retorna la fase actual de la onda."""
        return self.fase
    
    def get_amplitude(self) -> float:
        """Retorna la amplitud actual de la onda."""
        return self.amplitud
    
    def get_frequency(self) -> float:
        """Retorna la frecuencia actual de la onda."""
        return self.frecuencia
    
    def set_phase(self, nueva_fase: float) -> None:
        """Establece una nueva fase para la onda."""
        self.fase = nueva_fase
        logger.debug(f"Fase de '{self.nombre}' actualizada a: {nueva_fase} rad ({np.degrees(nueva_fase):.2f}°)")
    
    def set_amplitude(self, nueva_amplitud: float) -> None:
        """Establece una nueva amplitud para la onda."""
        self.amplitud = nueva_amplitud
        logger.debug(f"Amplitud de '{self.nombre}' actualizada a: {nueva_amplitud}")
    
    def set_frequency(self, nueva_frecuencia: float) -> None:
        """Establece una nueva frecuencia para la onda."""
        self.frecuencia = nueva_frecuencia
        logger.debug(f"Frecuencia de '{self.nombre}' actualizada a: {nueva_frecuencia}")
    
    def __str__(self) -> str:
        """Representación en cadena de la serie temporal."""
        return (f"TimeSeries('{self.nombre}': A={self.amplitud:.3f}, "
                f"f={self.frecuencia:.3f}, φ={self.fase:.3f} rad ({np.degrees(self.fase):.2f}°))")


# --- Función colapso_onda Mejorada ---
def colapso_onda(onda_superpuesta: np.ndarray, bayes_logic: BayesLogic, 
                prn_influence: float, previous_action: int) -> Tuple[float, int]:
    """
    Simula el colapso de la onda superpuesta basándose en la lógica Bayesiana.
    
    Args:
        onda_superpuesta (np.ndarray): Array numpy representando la onda superpuesta.
        bayes_logic (BayesLogic): Instancia de la clase BayesLogic para la toma de decisiones.
        prn_influence (float): Influencia del Ruido Probabilístico de Referencia (PRN).
        previous_action (int): Acción previa tomada (0 o 1).
    
    Returns:
        Tuple[float, int]: Estado colapsado (fase en radianes) y la acción seleccionada (0 o 1).
    """
    try:
        # 1. Calcular la entropía de la onda superpuesta
        entropy = shannon_entropy(onda_superpuesta)
        
        # 2. Definir un valor ambiental para calcular los cosenos directores
        env_value = 0.8
        
        # 3. Calcular los cosenos directores basados en la entropía y el valor ambiental
        cos_x, cos_y, cos_z = calculate_cosines(entropy, env_value)
        
        # 4. Calcular la coherencia basada en los cosenos directores
        coherence = (cos_x + cos_y + cos_z) / 3
        
        # 5. Utilizar BayesLogic para determinar la acción a tomar
        probabilities = bayes_logic.calculate_probabilities_and_select_action(
            entropy=entropy,
            coherence=coherence,
            prn_influence=prn_influence,
            action=previous_action
        )
        
        action_to_take = probabilities["action_to_take"]
        
        # 6. Simular el estado colapsado basándose en la acción
        if action_to_take == 1:
            # Añadir una pequeña variación aleatoria al colapso para hacer la simulación más interesante
            estado_colapsado = np.pi + (np.random.random() - 0.5) * 0.2
        else:
            estado_colapsado = 0.5 + (np.random.random() - 0.5) * 0.2
        
        logger.info(f"Colapso de onda - Entropía: {entropy:.4f}, Coherencia: {coherence:.4f}, "
                   f"Acción: {action_to_take}, Fase colapsada: {np.degrees(estado_colapsado):.2f}°")
        
        return estado_colapsado, action_to_take
    
    except Exception as e:
        logger.error(f"Error en colapso_onda: {e}")
        # Valores de fallback en caso de error
        return 0.0, previous_action


# --- Clase NeuralNetwork para gestionar la red neuronal ---
class NeuralNetwork:
    """
    Clase para gestionar una red neuronal simple.
    
    Attributes:
        layers (List[List[NeuralNode]]): Capas de nodos de la red neuronal.
        architecture (List[int]): Arquitectura de la red (número de nodos por capa).
    """
    def __init__(self, architecture: List[int]):
        """
        Inicializa la red neuronal con la arquitectura especificada.
        
        Args:
            architecture (List[int]): Lista con el número de nodos por capa.
        """
        self.architecture = architecture
        self.layers = []
        
        # Inicializar las capas y nodos
        for n_nodes in architecture:
            self.layers.append([initialize_node() for _ in range(n_nodes)])
        
        # Establecer conexiones entre capas adyacentes (ejemplo simplificado)
        for layer_idx in range(len(self.layers) - 1):
            for node_idx, node in enumerate(self.layers[layer_idx]):
                # Conectar con todos los nodos de la siguiente capa
                for next_node_idx in range(len(self.layers[layer_idx + 1])):
                    node.add_connection(layer_idx + 1, next_node_idx)
        
        logger.info(f"Red neuronal inicializada con arquitectura: {architecture}")
    
    def update_activations(self, action: int, activation_prob: float = 0.2) -> None:
        """
        Actualiza las activaciones de los nodos basándose en la acción.
        
        Args:
            action (int): Acción tomada (0 o 1).
            activation_prob (float): Probabilidad base de activación.
        """
        # Factor de aumento de probabilidad basado en la acción
        factor = 1.0 if action == 0 else 2.0
        
        # Actualizar activaciones aleatoriamente, pero con probabilidad influenciada por la acción
        for layer_idx, layer in enumerate(self.layers):
            for node_idx, node in enumerate(layer):
                # La probabilidad de activación aumenta en capas más profundas para acción 1
                layer_factor = 1.0 + (layer_idx / len(self.layers)) * (action * 0.5)
                
                if np.random.random() < (activation_prob * factor * layer_factor):
                    # Activar con un valor aleatorio entre 0.5 y 1.0
                    activation_value = 0.5 + np.random.random() * 0.5
                    node.activate(activation_value)
                else:
                    node.deactivate()
        
        logger.debug(f"Activaciones de la red actualizadas basadas en acción: {action}")
    
    def get_active_nodes_count(self) -> int:
        """Cuenta el número de nodos activos en la red."""
        return sum(1 for layer in self.layers for node in layer if node.is_active())
    
    def get_layers(self) -> List[List[NeuralNode]]:
        """Retorna las capas de la red neuronal."""
        return self.layers


# --- Funciones de visualización mejoradas ---
def visualize_wave_and_network(network: NeuralNetwork, iteration: int, t: float, 
                              ondas: Dict[str, TimeSeries], estado_colapsado_fase: float = None,
                              save_path: str = None) -> None:
    """
    Visualiza el estado de la red neuronal y la función de onda, incluyendo el estado colapsado.
    
    Args:
        network (NeuralNetwork): La red neuronal a visualizar.
        iteration (int): El número de iteración actual.
        t (float): El tiempo actual para la función de onda.
        ondas (Dict[str, TimeSeries]): Diccionario de ondas a visualizar.
        estado_colapsado_fase (float, opcional): La fase del estado colapsado para visualizar.
        save_path (str, opcional): Ruta para guardar la visualización como imagen.
    """
    try:
        # 1. Gráfico de la Función de Onda
        x_wave = np.linspace(0, 10, 500)
        
        plt.figure(figsize=(14, 8))
        
        # Subplot para la onda
        plt.subplot(1, 2, 1)
        
        # Graficar cada onda del diccionario
        for nombre, onda in ondas.items():
            if "Incidente" in nombre:
                plt.plot(x_wave, onda.evaluate(x_wave), label=nombre, color="blue", alpha=0.6)
            elif "Reflejada" in nombre:
                plt.plot(x_wave, onda.evaluate(x_wave), label=nombre, color="red", alpha=0.6)
            elif "Superpuesta" in nombre:
                plt.plot(x_wave, onda.evaluate(x_wave), label=nombre, color="green")
        
        # Graficar la onda colapsada si se proporciona
        if estado_colapsado_fase is not None:
            onda_colapsada = TimeSeries(
                amplitud=0.5, 
                frecuencia=1.5, 
                fase=estado_colapsado_fase,
                nombre="Onda Colapsada"
            )
            plt.plot(
                x_wave, 
                onda_colapsada.evaluate(x_wave), 
                label=f"Onda Colapsada (Fase: {np.degrees(estado_colapsado_fase):.0f}°)", 
                color="purple", 
                linestyle='--'
            )
        
        plt.xlabel("x")
        plt.ylabel("ψ(x)")
        plt.title(f"Superposición y Colapso de Ondas en t={t:.2f}")
        plt.grid(True)
        plt.legend()
        
        # 2. Gráfico del Estado de la Red Neuronal
        plt.subplot(1, 2, 2)
        
        # Obtener las capas de la red
        layers = network.get_layers()
        
        # Mostrar conexiones entre nodos (simplificado)
        for layer_idx, layer in enumerate(layers):
            for node_idx, node in enumerate(layer):
                # Dibujar conexiones a la siguiente capa
                if hasattr(node, 'connections') and layer_idx < len(layers) - 1:
                    for next_layer, next_node in node.connections:
                        if next_layer == layer_idx + 1:  # Solo mostrar conexiones a la siguiente capa
                            plt.plot(
                                [layer_idx, next_layer], 
                                [node_idx, next_node], 
                                color='gray', 
                                alpha=0.3, 
                                linewidth=0.5
                            )
        
        # Mostrar nodos
        for layer_idx, layer in enumerate(layers):
            for node_idx, node in enumerate(layer):
                if is_active(node):
                    # El tamaño del punto puede reflejar el valor de activación si es relevante
                    activation_size = 100 + 50 * node.activation_value if hasattr(node, 'activation_value') else 100
                    plt.scatter(layer_idx, node_idx, color='red', marker='o', s=activation_size)
                else:
                    plt.scatter(layer_idx, node_idx, color='blue', marker='o', s=50, alpha=0.3)
        
        plt.title(f"Estado de la Red Neuronal en Iteración {iteration}")
        plt.xlabel("Índice de Capa")
        plt.ylabel("Índice de Nodo")
        plt.xlim(-1, len(layers))
        plt.ylim(-1, max(len(layer) for layer in layers) if layers else 0)
        plt.grid(True)
        
        plt.tight_layout()
        
        # Guardar la figura si se proporciona una ruta
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualización guardada en: {save_path}")
        
        plt.show()
    
    except Exception as e:
        logger.error(f"Error en visualize_wave_and_network: {e}")


# --- Función principal para ejecutar la simulación ---
def main(seed: Optional[int] = None, save_path: str = "output/") -> None:
    """
    Función principal para ejecutar la simulación del colapso de onda y red neuronal.
    
    Args:
        seed (Optional[int]): Semilla para reproducibilidad (opcional).
        save_path (str): Ruta base para guardar resultados (opcional).
    """
    try:
        # Configurar semilla para reproducibilidad si se proporciona
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Simulación iniciada con semilla: {seed}")
        
        # Crear directorio para resultados si no existe
        os.makedirs(save_path, exist_ok=True)
        
        # Parámetros de las ondas
        amplitud = 0.5
        frecuencia = 1.5
        fase_inicial = -np.pi / 21
        
        # Crear las ondas iniciales
        onda_incidente = TimeSeries(amplitud, frecuencia, fase_inicial, nombre="Onda Incidente")
        onda_reflejada = TimeSeries(amplitud, frecuencia, fase_inicial + np.pi, nombre="Onda Reflejada")
        
        # Instanciar BayesLogic y PRN
        bayes_logic = BayesLogic()
        prn = PRN(influence=0.5, seed=seed)
        
        # Acción previa inicial
        previous_action = 1
        
        # Inicializar una red neuronal con una arquitectura específica
        network = NeuralNetwork([3, 4, 3, 2])
        
        # Guardar resultados para análisis posterior
        resultados = []
        
        # Número de iteraciones
        n_iteraciones = 10
        
        # Simulación iterativa
        for iteration in range(n_iteraciones):
            t = iteration * 0.5  # Tiempo para la función de onda
            
            # Generar puntos x para la visualización
            x = np.linspace(0, 10, 500)
            
            # Evaluar las ondas
            y_incidente = onda_incidente.evaluate(x)
            y_reflejada = onda_reflejada.evaluate(x)
            y_superpuesta = y_incidente + y_reflejada
            
            # Crear diccionario de ondas para visualización
            ondas = {
                "Onda Incidente": onda_incidente,
                "Onda Reflejada": onda_reflejada,
                "Onda Superpuesta": TimeSeries(1.0, frecuencia, 0, nombre="Onda Superpuesta")
            }
            
            # La onda superpuesta requiere un tratamiento especial
            ondas["Onda Superpuesta"].evaluate = lambda x_val: onda_incidente.evaluate(x_val) + onda_reflejada.evaluate(x_val)
            
            # Actualizar la influencia del PRN con una pequeña variación aleatoria
            prn.update_influence(prn.influence + (np.random.random() - 0.5) * 0.1)
            
            # Simular el colapso de la onda y obtener la acción y el estado colapsado
            estado_colapsado_fase, selected_action = colapso_onda(
                y_superpuesta, 
                bayes_logic, 
                prn.influence, 
                previous_action
            )
            
            # Establecer la fase de la onda incidente (como ejemplo de "influencia del colapso")
            onda_incidente.set_phase(estado_colapsado_fase)
            previous_action = selected_action  # Actualizar la acción previa para la siguiente iteración
            
            # Actualizar las activaciones de la red neuronal basadas en la acción
            network.update_activations(selected_action)
            
            # Ruta para guardar la visualización de esta iteración
            fig_path = os.path.join(save_path, f"iteracion_{iteration}.png")
            
            # Visualizar la onda y la red neuronal en cada iteración
            visualize_wave_and_network(
                network, 
                iteration, 
                t, 
                ondas, 
                estado_colapsado_fase,
                save_path=fig_path
            )
            
            # Calcular métricas para esta iteración
            entropy_val = shannon_entropy(y_superpuesta)
            cos_x_val, cos_y_val, cos_z_val = calculate_cosines(entropy_val, 0.8)
            coherence_val = (cos_x_val + cos_y_val + cos_z_val) / 3
            
            # Imprimir información de la iteración
            print(f"Iteración {iteration}:")
            print(f"  Entropía de Shannon: {entropy_val:.4f}")
            print(f"  Cosenos Directores: cos_x = {cos_x_val:.4f}, cos_y = {cos_y_val:.4f}, cos_z = {cos_z_val:.4f}")
            print(f"  Coherencia: {coherence_val:.4f}")
            print(f"  Influencia PRN: {prn.influence:.4f}")
            print(f"  Acción seleccionada: {'Mover Derecha' if selected_action == 1 else 'Mover Izquierda'}")
            print(f"  Estado colapsado (fase): {np.degrees(estado_colapsado_fase):.2f} grados")
            print(f"  Nodos activos: {network.get_active_nodes_count()} de {sum(network.architecture)}")
            print()
            
            # Guardar resultados para análisis posterior
            resultados.append({
                "iteracion": iteration,
                "tiempo": t,
                "entropia": entropy_val,
                "coherencia": coherence_val,
                "accion": selected_action,
                "fase_colapsada": estado_colapsado_fase,
                "nodos_activos": network.get_active_nodes_count()
            })
        
        # Guardar resultados como archivo numpy (opcional)
        np.save(os.path.join(save_path, "resultados.npy"), resultados)
        logger.info(f"Simulación completada. Resultados guardados en: {os.path.join(save_path, 'resultados.npy')}")
        
        # Visualización final del análisis de resultados
        if resultados:
            plt.figure(figsize=(12, 10))
            
            # Gráfico de evolución de la entropía
            plt.subplot(2, 2, 1)
            plt.plot([r["iteracion"] for r in resultados], [r["entropia"] for r in resultados], 'b-o')
            plt.title("Evolución de la Entropía")
            plt.xlabel("Iteración")
            plt.ylabel("Entropía")
            plt.grid(True)
            
            # Gráfico de evolución de la coherencia
            plt.subplot(2, 2, 2)
            plt.plot([r["iteracion"] for r in resultados], [r["coherencia"] for r in resultados], 'g-o')
            plt.title("Evolución de la Coherencia")
            plt.xlabel("Iteración")
            plt.ylabel("Coherencia")
            plt.grid(True)
            
            # Gráfico de acción seleccionada
            plt.subplot(2, 2, 3)
            plt.bar([r["iteracion"] for r in resultados], [r["accion"] for r in resultados])
            plt.title("Acción Seleccionada por Iteración")
            plt.xlabel("Iteración")
            plt.ylabel("Acción (0=Izq, 1=Der)")
            plt.ylim(-0.1, 1.1)
            plt.grid(True)
            
            # Gráfico de nodos activos
            plt.subplot(2, 2, 4)
            plt.plot([r["iteracion"] for r in resultados], [r["nodos_activos"] for r in resultados], 'r-o')
            plt.title("Evol