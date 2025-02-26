import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from typing import Callable, Tuple, Optional
from enum import Enum

class ActivationType(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"

class QuantumClassicalActivator:
    def __init__(self, n_qubits: int = 3):
        """
        Inicializa el sistema híbrido cuántico-clásico.
        
        Args:
            n_qubits: Número de qubits para el registro cuántico
        """
        self.n_qubits = n_qubits
        self._setup_activation_functions()
        
    def _setup_activation_functions(self):
        """Configura diccionarios de funciones de activación y sus derivadas."""
        self.activation_functions = {
            ActivationType.SIGMOID: lambda x: 1 / (1 + np.exp(-x)),
            ActivationType.TANH: lambda x: np.tanh(x),
            ActivationType.RELU: lambda x: np.maximum(0, x)
        }
        
        self.activation_derivatives = {
            ActivationType.SIGMOID: lambda x: x * (1 - x),
            ActivationType.TANH: lambda x: 1 - np.tanh(x)**2,
            ActivationType.RELU: lambda x: np.where(x > 0, 1, 0)
        }

    def create_quantum_circuit(self, activation_type: ActivationType) -> QuantumCircuit:
        """
        Crea un circuito cuántico que implementa una versión controlada 
        de la función de activación.
        
        Args:
            activation_type: Tipo de función de activación a implementar
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Parámetro para el ángulo de rotación
        theta = Parameter('θ')
        
        # Aplicar superposición controlada basada en el tipo de activación
        if activation_type == ActivationType.SIGMOID:
            # Implementación sigmoid-like usando rotaciones controladas
            qc.h(qr[0])  # Superposición inicial
            qc.cry(theta, qr[0], qr[1])  # Rotación controlada
            qc.cx(qr[1], qr[2])  # Entrelazamiento
            
        elif activation_type == ActivationType.TANH:
            # Implementación tanh-like usando fase controlada
            qc.h(qr[0])
            qc.cp(theta, qr[0], qr[1])
            qc.rz(theta/2, qr[1])
            
        elif activation_type == ActivationType.RELU:
            # Implementación ReLU-like usando medición controlada
            qc.x(qr[0])
            qc.ch(qr[0], qr[1])
            qc.measure_all()
            
        return qc

    def quantum_activated_forward(self, 
                                x: np.ndarray, 
                                activation_type: ActivationType,
                                collapse_threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Realiza una propagación hacia adelante combinando activación clásica
        con colapso cuántico controlado.
        
        Args:
            x: Array de entrada
            activation_type: Tipo de función de activación
            collapse_threshold: Umbral para el colapso cuántico
            
        Returns:
            Tuple con la salida activada y la probabilidad de colapso
        """
        # Activación clásica
        classical_activation = self.activation_functions[activation_type](x)
        
        # Calcular probabilidad de colapso basada en la activación
        collapse_probability = np.mean(np.abs(classical_activation))
        
        # Determinar si ocurre colapso cuántico
        if collapse_probability > collapse_threshold:
            # Colapso controlado: modificar la activación
            quantum_factor = np.sqrt(1 - collapse_probability)
            modified_activation = classical_activation * quantum_factor
        else:
            modified_activation = classical_activation
            
        return modified_activation, collapse_probability

    def hybrid_backpropagation(self,
                             gradient: np.ndarray,
                             activation_type: ActivationType,
                             collapse_probability: float) -> np.ndarray:
        """
        Implementa retropropagación híbrida que tiene en cuenta
        tanto la derivada clásica como el efecto del colapso cuántico.
        
        Args:
            gradient: Gradiente recibido de la capa siguiente
            activation_type: Tipo de función de activación usada
            collapse_probability: Probabilidad de colapso calculada en forward
            
        Returns:
            Gradiente modificado
        """
        # Obtener derivada clásica
        classical_derivative = self.activation_derivatives[activation_type]
        
        # Modificar el gradiente considerando el colapso cuántico
        quantum_factor = np.sqrt(1 - collapse_probability)
        modified_gradient = gradient * classical_derivative * quantum_factor
        
        return modified_gradient

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del activador híbrido
    hybrid_activator = QuantumClassicalActivator(n_qubits=3)
    
    # Datos de ejemplo
    x = np.array([0.5, -0.2, 0.8])
    
    # Realizar activación híbrida
    activated_output, collapse_prob = hybrid_activator.quantum_activated_forward(
        x, 
        ActivationType.SIGMOID,
        collapse_threshold=0.6
    )
    
    print(f"Entrada: {x}")
    print(f"Salida activada: {activated_output}")
    print(f"Probabilidad de colapso: {collapse_prob}")
    
    # Crear circuito cuántico correspondiente
    quantum_circuit = hybrid_activator.create_quantum_circuit(ActivationType.SIGMOID)
    print("\nCircuito cuántico generado:")
    print(quantum_circuit)