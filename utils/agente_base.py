"""
Módulo de Agentes Híbridos con Aprendizaje Cuántico

Este script define una clase base para diferentes tipos de agentes y sus respectivas subclases.
Cada tipo de agente demuestra una forma específica de enfoque de aprendizaje (supervisado y refuerzo),
integrando un componente cuántico avanzado a través de la clase QuantumState. Esta clase, a su vez,
hereda de QuantumBayesMahalanobis (definido en el módulo quantum_bayes_mahalanobis) para disponer de
métodos avanzados de proyección y predicción basados en cálculos de Mahalanobis y lógica bayesiana.

Autor: Jacobo Tlacaelel Mina Rodríguez  
Fecha: 13/03/2025  
Versión: cuadrante-coremind v1.0  
"""

import ast
import random
from typing import Dict, Any, Tuple
import numpy as np

# Se asume que el módulo 'quantum_bayes_mahalanobis' está disponible en el PATH
from quantum_bayes_mahalanobis import QuantumBayesMahalanobis
from bayes_logic import StatisticalAnalysis


# Clase QuantumState

class QuantumState(QuantumBayesMahalanobis):
    """
    Simula y gestiona un estado cuántico utilizando la lógica bayesiana
    y el cálculo de distancias de Mahalanobis (métodos heredados de 
    QuantumBayesMahalanobis).

    El estado se representa como un vector de probabilidades (normalizado)
    que se actualiza en cada interacción.
    """
    def __init__(self, num_positions: int, learning_rate: float = 0.1):
        """
        Inicializa el estado cuántico.

        Args:
            num_positions (int): Número de posiciones o dimensiones del estado.
            learning_rate (float): Tasa de actualización del estado.
        """
        super().__init__()  # Inicializa métodos de QuantumBayesMahalanobis
        self.num_positions = num_positions
        self.learning_rate = learning_rate
        # Se inicia con un estado aleatorio y se normaliza para representar probabilidades
        self.state_vector = np.random.rand(num_positions)
        self.state_vector = self.normalize_state(self.state_vector)
        self.probabilities = self.state_vector.copy()

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        """
        Normaliza el vector de estado para que su norma sea 1.

        Args:
            state (np.ndarray): Vector de estado.

        Returns:
            np.ndarray: Vector normalizado.
        """
        norm = np.linalg.norm(state)
        return state / norm if norm != 0 else state

    def predict_state_update(self) -> Tuple[np.ndarray, float]:
        """
        Predice el siguiente estado cuántico utilizando el método
        predict_quantum_state heredado de QuantumBayesMahalanobis.

        Se calculan valores de entropía (usando StatisticalAnalysis)
        y coherencia (por ejemplo, la media del vector) para alimentar
        la predicción.

        Returns:
            Tuple[np.ndarray, float]: (nuevo estado, valor posterior)
        """
        # Convertir el estado actual a forma adecuada (batch_size=1)
        input_state = np.array([self.state_vector])
        # Calcular entropía y coherencia
        entropy = StatisticalAnalysis.shannon_entropy(self.state_vector)
        coherence = np.mean(self.state_vector)
        next_state_tensor, posterior = self.predict_quantum_state(input_state, entropy, coherence)
        next_state = next_state_tensor.numpy().flatten()
        next_state = self.normalize_state(next_state)
        return next_state, posterior

    def update_state(self, action: int) -> None:
        """
        Actualiza el estado cuántico en función de una acción tomada.
        La actualización se realiza combinando el estado actual con la predicción
        generada, ponderada por la tasa de aprendizaje.

        Args:
            action (int): Acción ejecutada (por ejemplo, 0 o 1).
        """
        next_state, posterior = self.predict_state_update()
        self.state_vector = self.normalize_state(
            (1 - self.learning_rate) * self.state_vector +
            self.learning_rate * next_state
        )
        self.probabilities = self.state_vector.copy()

    def visualize_state(self) -> None:
        """
        Muestra en consola el estado cuántico (vector de probabilidades).
        """
        print("Estado cuántico (vector de probabilidades):")
        print(self.state_vector)



# Clase base de Agentes

class AgentBase:
    """
    Clase base para agentes, proporcionando mecánica e interacciones comunes
    con el componente cuántico.
    
    Attributes:
        name (str): Nombre del agente.
        quantum_state (QuantumState): Instancia para gestionar el estado cuántico.
    """
    def __init__(self, name: str, quantum_state: QuantumState):
        self.name = name
        self.quantum_state = quantum_state

    def interact(self, quantum_state_str: str, config_str: str) -> str:
        """
        Realiza una interacción tomando ambos parámetros, procesando sus configuraciones
        y utilizando el estado cuántico.

        Args:
            quantum_state_str (str): Estado cuántico en formato string (se espera parseable).
            config_str (str): String de configuración (se espera que incluya información clave).

        Returns:
            str: Mensaje de resultado de interacción.
        """
        # Se parsean los inputs de manera segura
        parsed_state, config = self._parse_input(quantum_state_str, config_str)
        # Se actualiza el estado cuántico a modo de ejemplo
        self.quantum_state.update_state(action=1)
        return f"{self.name} interactuó con la configuración: {config}"

    def _parse_input(self, quantum_state_str: str, config_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parsea de forma segura los strings de entrada usando ast.literal_eval.

        Args:
            quantum_state_str (str): Estado cuántico en formato string.
            config_str (str): Configuración en formato string.

        Returns:
            Tuple[Dict, Dict]: Tupla con los datos parseados.

        Raises:
            ValueError: Si no se pueden parsear los strings correctamente.
            NotImplementedError: Si faltan claves esperadas.
        """
        try:
            parsed_state = ast.literal_eval(quantum_state_str)
            parsed_config = ast.literal_eval(config_str)
            if 'nq_b' not in parsed_config:
                raise NotImplementedError("Se requiere la clave 'nq_b' en la configuración.")
            return parsed_state, parsed_config
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Error al parsear los datos: {e}")

    def learn(self) -> None:
        """
        Método base de aprendizaje a implementar por subclases.
        """
        raise NotImplementedError("Subclases deben implementar el método learn.")

# Agente Supervisado

class SupervisedAgent(AgentBase):
    """
    Agente que utiliza aprendizaje supervisado para tomar decisiones.
    
    Attributes:
        training_data (list): Datos de entrenamiento almacenados.
    """
    def __init__(self, name: str, quantum_state: QuantumState):
        super().__init__(name, quantum_state)
        self.training_data = []

    def interact(self, quantum_state_str: str, config_str: str) -> str:
        """
        Realiza una interacción supervisada, procesando la configuración y actualizando
        el aprendizaje.
        """
        result = super().interact(quantum_state_str, config_str)
        print(f"Agente supervisado {self.name}: {result}")
        self.learn()
        return result

    def learn(self) -> None:
        """
        Simula el aprendizaje supervisado actualizando internamente los datos de entrenamiento.
        """
        print(f"El agente supervisado {self.name} está aprendiendo...")
        self.training_data.append({"sample": "config", "output": "prediction"})
        print(f"Total muestras de entrenamiento: {len(self.training_data)}")

# Agente de Aprendizaje por Refuerzo

class ReinforcementAgent(AgentBase):
    """
    Agente que implementa aprendizaje por refuerzo, maximizando una recompensa acumulada.

    Attributes:
        rewards (int): Recompensa acumulada.
    """
    def __init__(self, name: str, quantum_state: QuantumState):
        super().__init__(name, quantum_state)
        self.rewards = 0

    def interact(self, quantum_state_str: str, config_str: str) -> str:
        """
        Realiza una interacción de refuerzo y actualiza la recompensa.
        """
        result = super().interact(quantum_state_str, config_str)
        self.learn()
        return result

    def learn(self) -> None:
        """
        Simula el aprendizaje por refuerzo seleccionando acciones al azar y
        actualizando la recompensa acumulada.
        """
        action = random.choice(["explore", "exploit"])
        reward = random.randint(-1, 1) if action == "explore" else 1
        self.rewards += reward
        print(f"Agente de refuerzo {self.name}: Acción: {action}, Recompensa: {reward}, "
              f"Recompensa total: {self.rewards}")

# Función principal de demostración

def main():
    """
    Función principal que demuestra el comportamiento de los agentes supervisado y
    de refuerzo, utilizando la integración de QuantumState para interactuar con
    configuraciones (strings) que contienen datos cuánticos.
    """
    # Ejemplo de strings de entrada (deben ser parseables)
    quantum_state_str = "{'state': '1:0'}"
    config_str = "{'nq_a':'value','nq_b':'otro_valor'}"

    # Inicializar el estado cuántico utilizando 2^num_qubits posiciones (por ejemplo, num_qubits=3 → 8 posiciones)
    qs = QuantumState(num_positions=8, learning_rate=0.1)

    # Crear instancias de agentes utilizando el mismo estado cuántico compartido
    supervised_agent = SupervisedAgent(name="SupervisedAgent1", quantum_state=qs)
    reinforcement_agent = ReinforcementAgent(name="ReinforcementAgent1", quantum_state=qs)

    print("\n--- Interacción con Agente Supervisado ---")
    supervised_agent.interact(quantum_state_str, config_str)

    print("\n--- Interacción con Agente de Refuerzo ---")
    reinforcement_agent.interact(quantum_state_str, config_str)

    print("\n--- Estado Cuántico Final ---")
    qs.visualize_state()

if __name__ == "__main__":
    main()