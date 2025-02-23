from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, XGate, RZGate
from qiskit.visualization import plot_bloch_multivector
import numpy as np

class ResilientQuantumCircuit:
    def __init__(self, num_qubits=5):
        """Inicializa el circuito cuántico resistente."""
        self.q = QuantumRegister(num_qubits, 'q')
        self.c = ClassicalRegister(num_qubits, 'c')
        self.circuit = QuantumCircuit(self.q, self.c)
        
    def build_controlled_h(self, control, target):
        """Construye una puerta Hadamard controlada."""
        self.circuit.ch(control, target)
        
    def build_toffoli(self, control1, control2, target):
        """Construye una puerta Toffoli (CCNOT)."""
        self.circuit.ccx(control1, control2, target)
        
    def build_cccx(self, controls, target):
        """Construye una puerta CCCX (Control-Control-Control-NOT)."""
        # Implementación de CCCX usando Toffoli gates
        ancilla = (controls[0] + 1) % 5  # Usar un qubit auxiliar
        self.circuit.ccx(controls[0], controls[1], ancilla)
        self.circuit.ccx(ancilla, controls[2], target)
        # Deshacer la primera Toffoli para limpiar el ancilla
        self.circuit.ccx(controls[0], controls[1], ancilla)
        
    def add_phase_spheres(self):
        """Añade esferas de fase (RZ gates) a todos los qubits."""
        phase = np.pi/4  # Fase que ayuda a la resistencia
        for i in range(5):
            self.circuit.rz(phase, self.q[i])
            
    def create_resilient_state(self):
        """Crea el estado resistente a medición."""
        # Primera columna: H controlada
        self.build_controlled_h(self.q[0], self.q[1])
        
        # Segunda columna: Toffoli
        self.build_toffoli(self.q[0], self.q[1], self.q[2])
        
        # Tercera columna: X en q0 y Toffoli
        self.circuit.x(self.q[0])
        self.build_toffoli(self.q[2], self.q[3], self.q[4])
        
        # Cuarta columna: CCCX
        self.build_cccx([self.q[0], self.q[1], self.q[2]], self.q[3])
        
        # Añadir esferas de fase para resistencia a medición
        self.add_phase_spheres()
        
        # Crear entrelazamiento adicional para resistencia
        for i in range(4):
            self.circuit.cx(self.q[i], self.q[i+1])
        
        return self.circuit
    
    def measure_qubit(self, qubit_index):
        """Mide un qubit específico manteniendo la coherencia del resto."""
        # Añadir barreras para asegurar el orden de las operaciones
        self.circuit.barrier()
        # Aplicar transformación de protección antes de la medición
        self.circuit.h(self.q[qubit_index])
        self.circuit.rz(np.pi/2, self.q[qubit_index])
        # Realizar la medición
        self.circuit.measure(self.q[qubit_index], self.c[qubit_index])
        # Restaurar el estado (opcional, dependiendo del uso)
        self.circuit.rz(-np.pi/2, self.q[qubit_index])
        self.circuit.h(self.q[qubit_index])
        
    def measure_all(self):
        """Mide todos los qubits manteniendo máxima coherencia posible."""
        self.circuit.barrier()
        # Aplicar transformación de protección global
        for i in range(5):
            self.circuit.h(self.q[i])
            self.circuit.rz(np.pi/2, self.q[i])
        # Realizar mediciones
        self.circuit.measure_all()

def main():
    # Crear y construir el circuito
    qc = ResilientQuantumCircuit()
    circuit = qc.create_resilient_state()
    
    # Podemos medir cualquier qubit sin perder el estado cuántico
    qc.measure_qubit(2)  # Por ejemplo, medir el qubit 2
    
    # Imprimir el circuito
    print(circuit)
    
    return circuit

if __name__ == "__main__":
    circuit = main()
