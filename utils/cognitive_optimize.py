import numpy as np
import matplotlib.pyplot as plt

class QuantumState:
    def __init__(self, num_positions, learning_rate=0.1):
        """
        Initialize QuantumState with enhanced tracking and learning capabilities.
        
        Args:
            num_positions (int): Number of possible quantum positions
            learning_rate (float, optional): Rate of probability update. Defaults to 0.1.
        """
        self.num_positions = num_positions
        self.learning_rate = learning_rate
        
        # Use more memory-efficient initialization methods
        self.probabilities = self._calculate_initial_probabilities()
        self.history = [self.probabilities.copy()]
        self.information_entropy = []
        self.information_history = []

    def _calculate_initial_probabilities(self):
        """
        Calculate initial probabilities with a uniform distribution.
        
        Returns:
            numpy.ndarray: Normalized probabilities
        """
        return np.ones(self.num_positions) / self.num_positions

    @property
    def angles(self):
        """
        Generate quantum state angles using non-uniform distribution.
        
        Returns:
            numpy.ndarray: Angles distributed between 0 and π
        """
        # Use more sophisticated angle distribution
        return np.linspace(0, np.pi, self.num_positions, endpoint=True)

    def update_probabilities(self, action):
        """
        Enhanced probability update with entropy tracking.
        
        Args:
            action (int): Action to update probabilities (0 or 1)
        """
        # More efficient probability update
        new_probabilities = self.probabilities.copy()
        observed_pos = self.observe_position()

        # Vectorized probability update
        direction_factor = 1 if action == 1 else -1
        proximity_adjustment = np.abs(np.arange(self.num_positions) - observed_pos)
        new_probabilities += direction_factor * self.learning_rate * (1 / (proximity_adjustment + 1))

        # Ensure non-negative and normalize probabilities
        new_probabilities = np.maximum(new_probabilities, 0)
        new_probabilities /= np.sum(new_probabilities)

        # Track entropy and information
        entropy = -np.sum(new_probabilities * np.log2(new_probabilities + 1e-10))
        info_gain = np.sum(np.abs(new_probabilities - self.probabilities))

        self.probabilities = new_probabilities
        self.history.append(new_probabilities.copy())
        self.information_entropy.append(entropy)
        self.information_history.append((action, info_gain))

    def observe_position(self):
        """
        Quantum state observation with probabilistic collapse.
        
        Returns:
            int: Observed quantum position
        """
        observed_position = np.random.choice(
            self.num_positions, 
            p=self.probabilities
        )
        return observed_position

    def visualize_state_evolution(self):
        """
        Advanced visualization of quantum state probabilities and entropy.
        """
        plt.figure(figsize=(15, 10))
        
        # Probability evolution heatmap
        plt.subplot(2, 2, 1)
        plt.imshow(np.array(self.history), aspect='auto', cmap='viridis')
        plt.title('Probability Evolution')
        plt.xlabel('Position')
        plt.ylabel('Time Step')
        plt.colorbar(label='Probability')
        
        # Information entropy plot
        plt.subplot(2, 2, 2)
        plt.plot(self.information_entropy)
        plt.title('Information Entropy Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        
        # Information gain plot
        plt.subplot(2, 2, 3)
        info_gains = [gain for _, gain in self.information_history]
        plt.plot(info_gains)
        plt.title('Information Gain')
        plt.xlabel('Time Step')
        plt.ylabel('Information Gain')
        
        # Final state probability distribution
        plt.subplot(2, 2, 4)
        plt.bar(range(self.num_positions), self.probabilities)
        plt.title('Final Probability Distribution')
        plt.xlabel('Position')
        plt.ylabel('Probability')
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
def main():
    num_qubits = 4  # Número de qubits
    num_positions = 2**num_qubits
    quantum_state = QuantumState(num_positions)

    # Simulación de evolución del estado cuántico
    for _ in range(50):
        action = np.random.randint(0, 2)
        quantum_state.update_probabilities(action)

    # Visualización del estado
    quantum_state.visualize_state_evolution()

if __name__ == "__main__":
    main()