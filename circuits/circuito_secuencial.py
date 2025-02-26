from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from math import log
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import minkowski
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class QubitsConfig:
    """Configuration parameters for quantum neural network"""
    activation_threshold: float = 0.5
    neighbor_distance_threshold: float = 0.7
    hidden_states: int = 1
    exposed_states: int = 2
    shots: int = 1024

class QuantumCircuitBuilder:
    """Handles creation and execution of quantum circuits"""
    
    @staticmethod
    def create_ccx_circuit() -> QuantumCircuit:
        """Create a Toffoli (CCX) gate circuit"""
        qc = QuantumCircuit(3, 3)
        qc.ccx(0, 1, 2)
        qc.measure_all()
        return qc
    
    @staticmethod
    def create_entangled_circuit() -> QuantumCircuit:
        """Create an entangled state circuit"""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
    
    @staticmethod
    def execute_circuit(circuit: QuantumCircuit, shots: int = 1024) -> Dict:
        """Execute a quantum circuit and return counts"""
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator, shots=shots)
        return job.result().get_counts(circuit)

class QuantumNode:
    """Represents a quantum node with quantum circuit integration"""
    def __init__(self, config: QubitsConfig):
        self.config = config
        self.state = self._initialize_state()
        self.circuit = self._initialize_circuit()
    
    def _initialize_state(self) -> np.ndarray:
        """Initialize quantum state"""
        total_size = self.config.exposed_states + self.config.hidden_states
        state = np.zeros((total_size, total_size), dtype=complex)
        
        for i in range(total_size):
            for j in range(total_size):
                state[i, j] = complex(np.random.rand(), np.random.rand())
        return state
    
    def _initialize_circuit(self) -> QuantumCircuit:
        """Initialize quantum circuit for the node"""
        qc = QuantumCircuit(3, 3)  # 3 qubits for CCX operations
        qc.h(0)  # Add superposition
        return qc
    
    def is_active(self) -> bool:
        """Determine node activity based on quantum measurement"""
        circuit = self.circuit.copy()
        circuit.measure_all()
        counts = QuantumCircuitBuilder.execute_circuit(circuit, self.config.shots)
        
        # Calculate activation based on measurement probabilities
        active_states = sum(counts[state] for state in counts 
                          if state.count('1') >= 2)  # At least 2 ones
        return (active_states / self.config.shots) > self.config.activation_threshold
    
    def apply_ccx_gate(self) -> None:
        """Apply Toffoli gate transformation"""
        ccx_circuit = QuantumCircuitBuilder.create_ccx_circuit()
        self.circuit.compose(ccx_circuit, inplace=True)
        
        # Update classical state representation
        self._update_state_from_circuit()
    
    def _update_state_from_circuit(self) -> None:
        """Update node state based on circuit execution"""
        counts = QuantumCircuitBuilder.execute_circuit(self.circuit)
        
        # Convert measurement outcomes to state amplitudes
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probability = count / total_shots
            self.state[idx // self.state.shape[1], idx % self.state.shape[1]] = \
                complex(np.sqrt(probability))

class QuantumNetwork:
    """Manages a network of quantum nodes with quantum circuit integration"""
    def __init__(
        self, 
        layer_sizes: List[int],
        config: Optional[QubitsConfig] = None
    ):
        self.config = config or QubitsConfig()
        self.network = [
            [QuantumNode(self.config) for _ in range(size)]
            for size in layer_sizes
        ]
        self.entanglement_circuit = QuantumCircuitBuilder.create_entangled_circuit()
    
    def calculate_neighbors(
        self,
        layer_idx: int,
        node_idx: int,
        p: int = 2
    ) -> int:
        """Calculate active neighbors using quantum measurements"""
        active_count = 0
        current_node = self.network[layer_idx][node_idx]
        
        for i in [layer_idx - 1, layer_idx + 1]:
            if 0 <= i < len(self.network):
                for neighbor in self.network[i]:
                    # Use quantum circuit to check entanglement
                    combined_circuit = self._create_neighbor_check_circuit(
                        current_node.circuit,
                        neighbor.circuit
                    )
                    counts = QuantumCircuitBuilder.execute_circuit(combined_circuit)
                    
                    # Check if nodes are quantum mechanically "close"
                    if self._are_nodes_entangled(counts) and neighbor.is_active():
                        active_count += 1
        
        return active_count
    
    def _create_neighbor_check_circuit(
        self,
        circuit1: QuantumCircuit,
        circuit2: QuantumCircuit
    ) -> QuantumCircuit:
        """Create a circuit to check quantum correlation between nodes"""
        qc = QuantumCircuit(4, 4)  # 2 qubits per node
        qc.compose(circuit1, qubits=[0, 1], inplace=True)
        qc.compose(circuit2, qubits=[2, 3], inplace=True)
        qc.measure_all()
        return qc
    
    def _are_nodes_entangled(self, counts: Dict) -> bool:
        """Determine if nodes are quantum mechanically correlated"""
        correlated_states = sum(counts[state] for state in counts 
                              if state.count('1') % 2 == 0)
        return (correlated_states / sum(counts.values())) > \
            self.config.neighbor_distance_threshold
    
    def evolve_network(self, iterations: int) -> None:
        """Evolve quantum network through iterations"""
        for iteration in range(iterations):
            for layer_idx, layer in enumerate(self.network):
                for node_idx, node in enumerate(layer):
                    active_neighbors = self.calculate_neighbors(layer_idx, node_idx)
                    self.apply_quantum_rules(node, active_neighbors)
            
            self.visualize(iteration)
            self._apply_network_entanglement()
    
    def apply_quantum_rules(
        self,
        node: QuantumNode,
        active_neighbors: int
    ) -> None:
        """Apply quantum-modified Conway rules"""
        if node.is_active():
            if active_neighbors < 2 or active_neighbors > 3:
                # Reset quantum state
                node.circuit = node._initialize_circuit()
                node._update_state_from_circuit()
        else:
            if active_neighbors == 3:
                # Activate through quantum operations
                node.apply_ccx_gate()
    
    def _apply_network_entanglement(self) -> None:
        """Apply entanglement operations across the network"""
        for layer in self.network:
            for i in range(len(layer) - 1):
                # Create entanglement between adjacent nodes
                combined_circuit = self._create_neighbor_check_circuit(
                    layer[i].circuit,
                    layer[i + 1].circuit
                )
                counts = QuantumCircuitBuilder.execute_circuit(combined_circuit)
                
                if self._are_nodes_entangled(counts):
                    layer[i].circuit.compose(self.entanglement_circuit, inplace=True)
                    layer[i + 1].circuit.compose(self.entanglement_circuit, inplace=True)
    
    def visualize(self, iteration: int) -> None:
        """Visualize quantum network state"""
        plt.figure(figsize=(12, 8))
        
        # Plot quantum nodes
        for layer_idx, layer in enumerate(self.network):
            for node_idx, node in enumerate(layer):
                # Get quantum state properties
                counts = QuantumCircuitBuilder.execute_circuit(node.circuit)
                total_shots = sum(counts.values())
                quantum_amplitude = max(count/total_shots for count in counts.values())
                
                color = 'blue' if node.is_active() else 'red'
                size = 200 * quantum_amplitude
                plt.scatter(layer_idx, node_idx, c=color, s=size, alpha=0.6)
        
        # Add quantum information
        entropy = self.calculate_network_entropy()
        plt.text(0.02, 0.98, f'Quantum Network Entropy: {entropy:.2f}',
                transform=plt.gca().transAxes)
        
        plt.title(f"Quantum Neural Network State - Iteration {iteration}")
        plt.xlabel("Layer Index")
        plt.ylabel("Node Index")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def calculate_network_entropy(self) -> float:
        """Calculate quantum entropy of the network"""
        probabilities = []
        for layer in self.network:
            for node in layer:
                counts = QuantumCircuitBuilder.execute_circuit(node.circuit)
                total_shots = sum(counts.values())
                probabilities.extend([count/total_shots for count in counts.values()])
        
        return -sum(p * log(p, 2) for p in probabilities if p > 0)

# Example usage
if __name__ == "__main__":
    config = QubitsConfig(
        activation_threshold=0.5,
        neighbor_distance_threshold=0.7,
        hidden_states=1,
        exposed_states=2,
        shots=1024
    )
    
    network = QuantumNetwork([1, 3, 1, 1], config)
    network.evolve_network(iterations=5)