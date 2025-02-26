import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, execute

class QuantumNeuron:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)
        self.simulator = Aer.get_backend('qasm_simulator')


    def build_circuit(self, inputs):
        """Builds the quantum circuit for the neuron.

        Args:
            inputs (list): List of input values (0 or 1) for each qubit.
        """

        if len(inputs) != self.num_qubits:
            raise ValueError("Number of inputs must match number of qubits.")
        
        self.circuit.reset(range(self.num_qubits))  # Start with a clean circuit

        for i, input_val in enumerate(inputs):
            if input_val == 1:
                self.circuit.x(i)  # Apply X gate if input is 1

        self.circuit.h(range(self.num_qubits)) # Apply Hadamard gates to all qubits
        self.circuit.measure_all() # Measure all


    def run_circuit(self, shots=1024):
        """Runs the quantum circuit and returns the counts.

        Args:
            shots (int): Number of times to run the simulation.

        Returns:
            dict: Dictionary of measurement counts.
        """
        job = execute(self.circuit, self.simulator, shots=shots)
        counts = job.result().get_counts()
        return counts

class QuantumState:
    def __init__(self, num_positions):
        """Initialize QuantumState with the number of possible positions."""
        self.num_positions = num_positions
        self.probabilities = self.calculate_initial_probabilities()
        self.history = [self.probabilities.copy()]

    def calculate_initial_probabilities(self):
        """Calculate initial probabilities (uniform distribution)."""
        initial_probabilities = np.ones(self.num_positions) / self.num_positions
        return initial_probabilities


# Example Usage 

num_qubits = 2  # Number of qubits for the quantum neuron
neuron = QuantumNeuron(num_qubits)

# The number of positions is determined by the number of qubits in the neuron
num_positions = 2**num_qubits  
quantum_state = QuantumState(num_positions)


# Example Loop :

for _ in range(5):
    # 1. Generate random input for the Quantum Neuron:
    input_values = np.random.randint(0, 2, size=num_qubits).tolist()

    # 2. Build and run the quantum circuit:
    neuron.build_circuit(input_values)
    counts = neuron.run_circuit()

    # 3. Get observed position from the Quantum Neuron's output:
    observed_state = max(counts, key=counts.get) # Classical outcome.
    observed_position = int(observed_state, 2)

    # 4. Now, USE the observed_position to update the QuantumState:
    quantum_state.update_probabilities(action=observed_position) 

    # ... methods and statics functions)
    def calculate_initial_probabilities(self):
        """
        Calculate initial probabilities based on a uniform distribution.
        """
        initial_probabilities = np.ones(self.num_positions) / self.num_positions  # Uniform distribution
        return initial_probabilities
    def calculate_num_positions(self):
        """
        Calculate the number of positions based on capas and estado.

        Returns:
            int: Number of positions (product of capas and estado length).
        """
        return len(self.estado) * self.capas

    def __init__(self, quantum_neuron: QuantumNeuron):
        """
        Initialize the QuantumState class with a QuantumNeuron instance.
        Calculate num_positions based on the neuron and initialize state.

        Args:
            quantum_neuron (QuantumNeuron): Instance of QuantumNeuron.
        """
        self.num_positions = quantum_neuron.calculate_num_positions()
        self.history = [self.probabilities.copy()]  # To track probability updates over time
    
    @property
    def angles(self):
        """
        Property to calculate and return angles distributed between 0 and π.
        
        Returns:
            numpy.ndarray: Array of angles.
        """
        return np.linspace(0, np.pi, self.num_positions)
    
    @property
    def probabilities(self):
        """
        Property to calculate and return normalized probabilities based on squared cosines.
        
        Returns:
            numpy.ndarray: Normalized probabilities for each position.
        """
        cosines = np.cos(self.angles)
        probabilities = cosines**2
        return probabilities / np.sum(probabilities)

    def calculate_probabilities(self):
        """
        Calculate initial probabilities using squared cosines of the angles.

        Returns:
            numpy.ndarray: Normalized probabilities for each position.
        """
        cosines = np.cos(self.angles)  # Compute cosine values for each angle
        probabilities = cosines**2  # Square the cosines to get positive values
        return probabilities / np.sum(probabilities)  # Normalize so sum of probabilities is 1

    def update_probabilities(self, action, k=0.1):
        """
        Update the probabilities based on the given action.

        Args:
            action (int): 0 for moving left, 1 for moving right.
            k (float): Scaling factor for probability adjustments.
        """
        new_probabilities = self.probabilities.copy()
        for i in range(self.num_positions):
            if action == 1:  # Action to move right
                if i > self.observe_position():
                    # Increase probability if position is to the right
                    new_probabilities[i] += k * self.probabilities[self.observe_position()]
                elif i < self.observe_position():
                    # Decrease probability if position is to the left
                    new_probabilities[i] -= k * self.probabilities[self.observe_position()]
                else:
                    # Increase probability significantly if it matches the observed position
                    new_probabilities[i] += (self.num_positions - 1) * k * self.probabilities[self.observe_position()]
            elif action == 0:  # Action to move left
                if i < self.observe_position():
                    # Increase probability if position is to the left
                    new_probabilities[i] += k * self.probabilities[self.observe_position()]
                elif i > self.observe_position():
                    # Decrease probability if position is to the right
                    new_probabilities[i] -= k * self.probabilities[self.observe_position()]
                else:
                    # Increase probability significantly if it matches the observed position
                    new_probabilities[i] += (self.num_positions - 1) * k * self.probabilities[self.observe_position()]

        # Normalize probabilities to ensure they sum to 1
        new_probabilities = new_probabilities / np.sum(new_probabilities)

        self.history.append(new_probabilities.copy())  # Save the updated probabilities to history

    def observe_position(self):
        """
        Observes the position, causing wave function collapse.
    
        Returns:
            int: The observed position.
        """
        observed_position = np.random.choice(self.num_positions, p=self.probabilities)
        new_probabilities = np.zeros(self.num_positions)
        new_probabilities[observed_position] = 1  # Collapse to the observed state
        self.history.append(new_probabilities.copy())  # Save collapsed state in history
        return observed_position
    

    def get_probabilities(self):
        """
        Get the current probabilities.
    
        Returns:
            numpy.ndarray: Current probabilities.
        """
        return self.probabilities
    def plot_probabilities(self):
        """Prints a text-based representation of the probability evolution."""
        print("Probability Evolution Over Time")
        [print(f"Position {i}: {[round(state[i], 3) for state in self.history]}") for i in range(self.num_positions)]

Example Usage (Integrating QuantumNeuron and QuantumState)

num_qubits = 2  # Number of qubits for the quantum neuron
neuron = QuantumNeuron(num_qubits)

num_positions = 2**num_qubits # Number of positions based on the number of qubits
quantum_state = QuantumState(num_positions)

# Example of running the quantum neuron and updating probabilities based on the outcome

input_values = [0, 1]  # Example inputs to the quantum neuron
neuron.build_circuit(input_values)
counts = neuron.run_circuit()

# Process the counts to get the observed position (most frequent measurement)
observed_state = max(counts, key=counts.get)
observed_position = int(observed_state, 2)  # Convert binary string to integer

# Update quantum state based on the observation
quantum_state.update_probabilities(action=observed_position)  # Use observed position as action.
quantum_state.plot_probabilities()


# Example loop incorporating the neuron
for _ in range(5): # Reduced number of iterations for the example
    input_values = np.random.randint(0, 2, size=num_qubits).tolist() # Random input values
    neuron.build_circuit(input_values)
    counts = neuron.run_circuit()
    observed_state = max(counts, key=counts.get)
    observed_position = int(observed_state, 2)

    quantum_state.update_probabilities(action=np.random.randint(0, 2)) # Random action for demonstration
    quantum_state.plot_probabilities()
    print(f"Observed Position (from Quantum Neuron): {observed_position}") # Print observation
    print(f"Observed position (from Quantum State): {quantum_state.observe_position()}")