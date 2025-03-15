"""
──────────────────────────────────────────────
Documentación del Script "QuantumState" y Simulación de Evolución

Este script implementa la clase QuantumState, que simula un estado cuántico mediante una distribución de probabilidades. El estado se actualiza con base en una acción observada (por ejemplo, 0 o 1), lo que provoca un ajuste en la distribución mediante un mecanismo basado en la proximidad del valor observado. Además, se registran métricas de información (entropía e información ganada) y se provee una función para visualizar la evolución del estado a lo largo del tiempo.

Autor: Jacobo Tlacaelel Mina Rodríguez  
Fecha: 13/03/2025  
Versión: cuadrante-coremind v1.0

────────────────────────────────────────────
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging

class QuantumState:
    def __init__(
        self, 
        num_positions: int, 
        learning_rate: float = 0.1, 
        random_seed: Optional[int] = None
    ):
        """
        Initialize a probabilistic quantum state simulation.

        Args:
            num_positions (int): Total number of possible quantum positions.
            learning_rate (float, optional): Learning rate for probability updates. Defaults to 0.1.
            random_seed (int, optional): Seed for reproducibility. Defaults to None.
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Logging configuration
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # State parameters
        self.num_positions = num_positions
        self.learning_rate = learning_rate

        # Initialize state tracking
        self.probabilities = self._initialize_probabilities()
        self.history: List[np.ndarray] = [self.probabilities.copy()]
        self.information_entropy: List[float] = []
        self.information_history: List[Tuple[int, float]] = []

        # Log initialization
        self.logger.info(f"Quantum State initialized with {num_positions} positions")

    def _initialize_probabilities(self) -> np.ndarray:
        """
        Create initial probability distribution.

        Returns:
            np.ndarray: Uniform probability distribution
        """
        return np.full(self.num_positions, 1.0 / self.num_positions)

    @property
    def angles(self) -> np.ndarray:
        """
        Generate quantum state angles using non-uniform distribution.

        Returns:
            np.ndarray: Angles distributed between 0 and π
        """
        return np.linspace(0, np.pi, self.num_positions, endpoint=True)

    def update_probabilities(self, action: int) -> None:
        """
        Update quantum state probabilities based on observed action.

        Args:
            action (int): Action influencing probability distribution (0 or 1)
        """
        try:
            observed_pos = self.observe_position()

            # Compute proximity-based adjustment
            direction_factor = 1 if action == 1 else -1
            proximity_mask = np.abs(np.arange(self.num_positions) - observed_pos)
            adjustment = direction_factor * self.learning_rate / (proximity_mask + 1)

            # Update probabilities with safeguards
            new_probabilities = np.clip(
                self.probabilities + adjustment, 
                0, 
                None  # No upper bound
            )
            new_probabilities /= new_probabilities.sum()  # Normalize

            # Compute information metrics
            entropy = self._compute_entropy(new_probabilities)
            info_gain = np.sum(np.abs(new_probabilities - self.probabilities))

            # Update state
            self.probabilities = new_probabilities
            self.history.append(new_probabilities.copy())
            self.information_entropy.append(entropy)
            self.information_history.append((action, info_gain))

        except Exception as e:
            self.logger.error(f"Error in probability update: {e}")
            raise

    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy of the probability distribution.

        Args:
            probabilities (np.ndarray): Probability distribution

        Returns:
            float: Entropy value
        """
        # Add small epsilon to prevent log(0)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def observe_position(self) -> int:
        """
        Simulate quantum state observation with probabilistic collapse.

        Returns:
            int: Observed quantum position
        """
        return np.random.choice(self.num_positions, p=self.probabilities)

    def visualize_state_evolution(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of quantum state evolution.

        Args:
            save_path (str, optional): Path to save visualization. Defaults to None.
        """
        plt.figure(figsize=(16, 12))
        plt.suptitle('Quantum State Evolution Metrics', fontsize=16)

        # Probability evolution heatmap
        plt.subplot(2, 2, 1)
        plt.imshow(np.array(self.history), aspect='auto', cmap='viridis')
        plt.title('Probability Evolution')
        plt.xlabel('Position')
        plt.ylabel('Time Step')
        plt.colorbar(label='Probability')

        # Information entropy plot
        plt.subplot(2, 2, 2)
        plt.plot(self.information_entropy, label='Entropy')
        plt.title('Information Entropy Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        plt.legend()

        # Information gain plot
        plt.subplot(2, 2, 3)
        info_gains = [gain for _, gain in self.information_history]
        plt.plot(info_gains, label='Information Gain', color='orange')
        plt.title('Information Gain')
        plt.xlabel('Time Step')
        plt.ylabel('Gain')
        plt.legend()

        # Final state probability distribution
        plt.subplot(2, 2, 4)
        plt.bar(range(self.num_positions), self.probabilities)
        plt.title('Final Probability Distribution')
        plt.xlabel('Position')
        plt.ylabel('Probability')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

def simulate_quantum_state(
    num_qubits: int = 4, 
    num_iterations: int = 50, 
    random_seed: Optional[int] = None
) -> QuantumState:
    """
    Simulate quantum state evolution.

    Args:
        num_qubits (int, optional): Number of qubits. Defaults to 4.
        num_iterations (int, optional): Number of evolution iterations. Defaults to 50.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        QuantumState: Evolved quantum state
    """
    num_positions = 2**num_qubits
    quantum_state = QuantumState(num_positions, random_seed=random_seed)

    for _ in range(num_iterations):
        action = np.random.randint(0, 2)
        quantum_state.update_probabilities(action)

    return quantum_state

def main():
    # Simulate and visualize quantum state
    quantum_state = simulate_quantum_state(num_qubits=4, random_seed=42)
    quantum_state.visualize_state_evolution(save_path='quantum_state_evolution.png')

if __name__ == "__main__":
    main()