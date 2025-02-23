from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import HGate, XGate, RZGate
import numpy as np
from qiskit_aer import AerSimulator

# Assuming ResilientQuantumCircuit class is defined in circuito_principal.py
from circuito_principal import ResilientQuantumCircuit

simulator = AerSimulator(method='statevector') # Define simulator globally for efficiency

def apply_action_and_get_state(current_circuit, action, current_phase):
    """
    Aplica una acción RL al circuito cuántico y simula para obtener el siguiente estado.

    Args:
        current_circuit (QuantumCircuit): El circuito cuántico actual.
        action (int): La acción RL a ejecutar (0: decrease phase, 1: maintain, 2: increase phase).
        current_phase (float): The current phase value of the RZ gates.

    Returns:
        np.array: El statevector del circuito modificado después de la simulación.
        float: The updated phase value.
    """
    delta_phase_step = np.pi / 16  # Define the phase step size
    updated_phase = current_phase # Initialize with current phase

    if action == 0: # Decrease phase
        updated_phase -= delta_phase_step
    elif action == 2: # Increase phase
        updated_phase += delta_phase_step
    # Action 1: Maintain phase - no change needed

    # Create a new circuit based on the current circuit (to avoid modifying the original)
    modified_circuit = QuantumCircuit(*current_circuit.qregs, *current_circuit.cregs) # Create a new circuit with same registers
    for instruction in current_circuit.data: # Copy instructions
        modified_circuit.append(instruction)

    # Modify RZ gates in the *new* circuit with the updated phase
    rz_gates_indices = [i for i, instruction in enumerate(modified_circuit.data) if isinstance(instruction[0], RZGate)] # Find indices of RZ gates
    for index in rz_gates_indices:
        qubit_to_apply_rz = modified_circuit.data[index][1][0] # Get the qubit RZ gate is applied to
        modified_circuit.data[index] = (RZGate(updated_phase), [qubit_to_apply_rz], []) # Replace with new RZ gate with updated phase

    # Simulate the modified circuit
    compiled_circuit = transpile(modified_circuit, simulator)
    job = simulator.run(compiled_circuit)
    result = job.result()
    next_statevector = np.array(result.get_statevector(compiled_circuit))

    return next_statevector, updated_phase


# --- Example Usage (for testing apply_action_and_get_state) ---
if __name__ == "__main__":
    qc = ResilientQuantumCircuit()
    initial_circuit = qc.create_resilient_state()
    initial_phase = np.pi/4 # Initial phase in create_resilient_state

    # Example action: Increase phase (action=2)
    action_to_apply = 2
    next_state, updated_phase_val = apply_action_and_get_state(initial_circuit, action_to_apply, initial_phase)

    print("Original Circuit:")
    print(initial_circuit.draw(output='text'))
    print("\nModified Circuit (after action):")
    # Note: Drawing the modified_circuit directly might not reflect in-place RZ gate change visually in some Qiskit versions.
    # To verify, you'd need to inspect the circuit data directly.
    modified_qc = ResilientQuantumCircuit() # Create a new ResilientQuantumCircuit instance
    modified_circuit_example = modified_qc.create_resilient_state() # Get a base circuit
    rz_gates_indices_example = [i for i, instruction in enumerate(modified_circuit_example.data) if isinstance(instruction[0], RZGate)]
    for index in rz_gates_indices_example: # Manually update phase for visualization example
        qubit_to_apply_rz = modified_circuit_example.data[index][1][0]
        modified_circuit_example.data[index] = (RZGate(updated_phase_val), [qubit_to_apply_rz], [])
    print(modified_circuit_example.draw(output='text'))


    print("\nNext Statevector (after action):")
    print(next_state)
    print("\nUpdated Phase Value:", updated_phase_val)
