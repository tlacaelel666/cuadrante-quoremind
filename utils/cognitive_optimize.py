"""
Módulo de Simulación de Estado Cuántico

Este módulo proporciona una implementación de un modelo simplificado 
de evolución de estado cuántico con capacidades de aprendizaje y visualización.

Características principales:
- Simulación de distribución de probabilidades cuánticas
- Seguimiento de entropía de información
- Visualización de la evolución del estado

Autor: Jacobo Tlacaelel Mina Rodríguez 
Fecha: 13 marzo 2025
Versión: cuadrante-coremind v1.0
"""
# cognitive_optimize.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional

class QuantumState:
    """
    Clase para simular la evolución de un estado cuántico probabilístico.

   # Atributos:
    ----------
    num_positions : int
        Número de posiciones posibles en el estado cuántico
    learning_rate : float
        Tasa de aprendizaje para actualizar probabilidades
    probabilities : numpy.ndarray
        Distribución de probabilidades actual
    history : List[numpy.ndarray]
        Historial de distribuciones de probabilidades
    information_entropy : List[float]
        Registro de entropía de información
    information_history : List[Tuple[int, float]]
        Historial de acciones e información ganada

   # Métodos:
    --------
    
update_probabilities(action: int)
        Actualiza las probabilidades basado en una acción
    observe_position() -> int
        Observa una posición basada en la distribución de probabilidades
    visualize_state_evolution()
        Visualiza la evolución del estado cuántico
    """

    def __init__(self, 
                 num_positions: int, 
                 learning_rate: float = 0.1, 
                 seed: Optional[int] = None):
        """
        Inicializa el estado cuántico con parámetros configurables.

        Parámetros:
        -----------
        num_positions : int
            Número de posiciones posibles en el estado cuántico
        learning_rate : float, opcional
            Tasa de aprendizaje para actualizar probabilidades (defecto: 0.1)
        seed : int, opcional
            Semilla para reproducibilidad de resultados aleatorios
        
      # Raises:
        -------
        ValueError
            Si los parámetros de entrada no son válidos
        """
        # Validación de parámetros de entrada
        if num_positions <= 0:
            raise ValueError("El número de posiciones debe ser positivo")
        if not 0 < learning_rate <= 1:
            raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1")
        
        # Configuración de semilla aleatoria
        if seed is not None:
            np.random.seed(seed)
        
        # Inicialización de atributos
        self.num_positions = num_positions
        self.learning_rate = np.float32(learning_rate)
        
        # Inicialización de probabilidades
        self.probabilities = self._calculate_initial_probabilities()
        
        # Inicialización de historiales
        self.history: List[np.ndarray] = [self.probabilities.copy()]
        self.information_entropy: List[float] = []
        self.information_history: List[Tuple[int, float]] = []

    def _calculate_initial_probabilities(self) -> np.ndarray:
        """
        Calcula las probabilidades iniciales con distribución uniforme.

        Returns:
        --------
        numpy.ndarray
            Probabilidades normalizadas iniciales
        """
        return np.ones(self.num_positions, dtype=np.float32) / self.num_positions

    def update_probabilities(self, action: int) -> None:
        """
        Actualiza las probabilidades del estado cuántico.

       # Parámetros:
        -----------
        action : int
            Acción para actualizar probabilidades (0 o 1)
        
       # Notas:
        ------
        - Actualiza probabilidades basado en la acción y posición observada
        - Calcula entropía e información ganada
        """
        # Copia de probabilidades actuales
        new_probabilities = self.probabilities.copy()
        
        # Observación de posición
        observed_pos = self.observe_position()

        # Actualización vectorizada de probabilidades
        direction_factor = 1 if action == 1 else -1
        proximity_adjustment = np.abs(np.arange(self.num_positions) - observed_pos)
        new_probabilities += direction_factor * self.learning_rate * (1 / (proximity_adjustment + 1))

        # Normalización de probabilidades
        new_probabilities = np.maximum(new_probabilities, 0)
        new_probabilities /= np.sum(new_probabilities)

        # Cálculo de entropía e información
        epsilon = np.finfo(float).eps
        entropy = -np.sum(new_probabilities * np.log2(new_probabilities + epsilon))
        info_gain = np.sum(np.abs(new_probabilities - self.probabilities))

        # Actualización de estado
        self.probabilities = new_probabilities
        self.history.append(new_probabilities.copy())
        self.information_entropy.append(entropy)
        self.information_history.append((action, info_gain))

    def observe_position(self) -> int:
        """
        Observa una posición basada en la distribución de probabilidades actual.

       # Returns:
        --------
        int
            Posición cuántica observada
        """
        return np.random.choice(
            self.num_positions, 
            p=self.probabilities
        )

    def visualize_state_evolution(self) -> None:
        """
        Visualiza la evolución del estado cuántico con múltiples gráficos.
        
      # Gráficos:
        ---------
        1. Mapa de calor de evolución de probabilidades
        2. Entropía de información
        3. Ganancia de información
        4. Distribución final de probabilidades
        """
        # Configuración de estilo
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 12))
        plt.suptitle('Evolución del Estado Cuántico', fontsize=16)
        
        # Mapa de calor de probabilidades
        plt.subplot(2, 2, 1)
        sns.heatmap(np.array(self.history), cmap='viridis', 
                    cbar_kws={'label': 'Probabilidad'})
        plt.title('Evolución de Probabilidades')
        plt.xlabel('Posición')
        plt.ylabel('Paso de Tiempo')
        
        # Entropía de información
        plt.subplot(2, 2, 2)
        plt.plot(self.information_entropy, color='blue')
        plt.title('Entropía de Información')
        plt.xlabel('Paso de Tiempo')
        plt.ylabel('Entropía')
        
        # Ganancia de información
        plt.subplot(2, 2, 3)
        info_gains = [gain for _, gain in self.information_history]
        plt.plot(info_gains, color='green')
        plt.title('Ganancia de Información')
        plt.xlabel('Paso de Tiempo')
        plt.ylabel('Ganancia')
        
        # Distribución final de probabilidades
        plt.subplot(2, 2, 4)
        plt.bar(range(self.num_positions), self.probabilities, color='red')
        plt.title('Distribución Final de Probabilidades')
        plt.xlabel('Posición')
        plt.ylabel('Probabilidad')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Función principal para demostrar el uso de QuantumState.
    
    Realiza una simulación de evolución de estado cuántico 
    y visualiza los resultados.
    """
    try:
        # Configuración de parámetros
        num_qubits = 4
        num_positions = 2**num_qubits
        
        # Creación del estado cuántico
        quantum_state = QuantumState(num_positions, seed=42)

        # Simulación de evolución
        for i in range(50):
            # Selección de acción con distribución no uniforme
            action = np.random.choice([0, 1], p=[0.6, 0.4])
            quantum_state.update_probabilities(action)
            
            # Información de progreso
            if i % 10 == 0:
                print(f"Iteración {i}: Entropía = {quantum_state.information_entropy[-1]:.4f}")

        # Visualización final
        quantum_state.visualize_state_evolution()
    
    except Exception as e:
        print(f"Error en la simulación: {e}")

if __name__ == "__main__":
    main()