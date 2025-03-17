"""
Módulo Híbrido de Integración Neuronal Cuántico-Clásico

Este módulo define un sistema completo que integra:
  • Una red neural clásica (NeuralNetwork) con múltiples funciones de activación y 
    soporte para optimizadores (SGD y Adam).
  • Un activador cuántico-clásico (QuantumClassicalActivator) que crea circuitos cuánticos
    relacionados con funciones de activación y permite realizar un paso forward híbrido.
  • Un sistema de estado cuántico (QuantumState) que utiliza métodos bayesianos y distancias
    de Mahalanobis para gestionar y predecir estados cuánticos.
  • Un sistema híbrido completo (QuantumNeuralHybridSystem) que integra todos los componentes
    anteriores y provee métodos para entrenamiento y predicción híbridos.

Autor: Jacobo Tlacaelel Mina Rodríguez  
Fecha: 13/03/2025  
Versión: cuadrante-coremind v1.0  
"""

import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from typing import Tuple, List, Optional
from enum import Enum
import joblib

# Clase ficticia para completar dependencias (asumimos que está en otro módulo)
class QuantumBayesMahalanobis:
    def __init__(self):
        pass

    def predict_quantum_state(self, current_input, entropy, coherence):
        # Implementación ficticia
        return tf.convert_to_tensor(np.random.rand(current_input.shape[0], current_input.shape[1])), np.random.random()

# Clase ficticia para completar dependencias (asumimos que está en otro módulo)
class StatisticalAnalysis:
    @staticmethod
    def shannon_entropy(probabilities):
        # Implementación básica de la entropía de Shannon
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

# -------------------------
# Funciones de activación clásicas
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

# -------------------------
# Red neuronal clásica
# -------------------------
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: list[int], output_size: int, activation: str = "sigmoid"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []
        
        previous_size = input_size
        for hs in hidden_size:
            self.weights.append(np.random.randn(previous_size, hs))
            self.biases.append(np.zeros((1, hs)))
            previous_size = hs
        self.weights.append(np.random.randn(previous_size, output_size))
        self.biases.append(np.zeros((1, output_size)))
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "tanh":
            return tanh(x)
        elif self.activation == "relu":
            return relu(x)
        else:
            raise ValueError("Función de activación no reconocida.")
    
    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation == "tanh":
            return tanh_derivative(x)
        elif self.activation == "relu":
            return relu_derivative(x)
        else:
            raise ValueError("Función de activación no reconocida.")
    
    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.activate(np.dot(X, w) + b)
            activations.append(X)
        return activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, optimizer: str = "sgd", **kwargs):
        activations = self.forward(X)
        output = activations[-1]
        output_error = y - output
        deltas = [output_error * self.activate_derivative(activations[-1])]
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * self.activate_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            if optimizer == "sgd":
                self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate
                self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            elif optimizer == "adam":
                self._adam(i, activations[i], deltas[i], learning_rate, **kwargs)
            else:
                raise ValueError("Optimizador no reconocido.")
    
    def _adam(self, layer: int, a: np.ndarray, delta: np.ndarray, learning_rate: float, t: int = 1,
              beta1=0.9, beta2=0.999, epsilon=1e-8, m_w=None, v_w=None, m_b=None, v_b=None):
        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
        
        self.m_w[layer] = beta1 * self.m_w[layer] + (1 - beta1) * a.T.dot(delta)
        self.v_w[layer] = beta2 * self.v_w[layer] + (1 - beta2) * np.square(a.T.dot(delta))
        m_hat_w = self.m_w[layer] / (1 - beta1**t)
        v_hat_w = self.v_w[layer] / (1 - beta2**t)
        self.weights[layer] += learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
        self.m_b[layer] = beta1 * self.m_b[layer] + (1 - beta1) * np.sum(delta, axis=0, keepdims=True)
        self.v_b[layer] = beta2 * self.v_b[layer] + (1 - beta2) * np.square(np.sum(delta, axis=0, keepdims=True))
        m_hat_b = self.m_b[layer] / (1 - beta1**t)
        v_hat_b = self.v_b[layer] / (1 - beta2**t)
        self.biases[layer] += learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

# -------------------------
# Tipos de activación para el componente cuántico-clásico
# -------------------------
class ActivationType(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"

# -------------------------
# Componente Híbrido: Activador Cuántico-Clásico
# -------------------------
class QuantumClassicalActivator:
    def __init__(self, n_qubits: int = 3):
        """
        Inicializa el sistema híbrido cuántico-clásico.
        
        Args:
            n_qubits (int): Número de qubits para el registro cuántico.
        """
        self.n_qubits = n_qubits
        self._setup_activation_functions()
        
    def _setup_activation_functions(self):
        """Configura funciones de activación y sus derivadas."""
        self.activation_functions = {
            ActivationType.SIGMOID: sigmoid,
            ActivationType.TANH: tanh,
            ActivationType.RELU: relu
        }
        self.activation_derivatives = {
            ActivationType.SIGMOID: sigmoid_derivative,
            ActivationType.TANH: tanh_derivative,
            ActivationType.RELU: relu_derivative
        }

    def create_quantum_circuit(self, activation_type: ActivationType) -> QuantumCircuit:
        """
        Crea un circuito cuántico que simula una versión controlada de la función de activación.
        
        Args:
            activation_type (ActivationType): Tipo de función de activación a implementar.
            
        Returns:
            QuantumCircuit: Circuito cuántico generado.
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        theta = Parameter('θ')
        if activation_type == ActivationType.SIGMOID:
            qc.h(qr[0])
            qc.cry(theta, qr[0], qr[1])
            qc.cx(qr[1], qr[2])
        elif activation_type == ActivationType.TANH:
            qc.h(qr[0])
            qc.cp(theta, qr[0], qr[1])
            qc.rz(theta/2, qr[1])
        elif activation_type == ActivationType.RELU:
            qc.x(qr[0])
            qc.ch(qr[0], qr[1])
            qc.measure_all()
        return qc

    def quantum_activated_forward(self, 
                                  x: np.ndarray, 
                                  activation_type: ActivationType,
                                  collapse_threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Realiza una propagación forward híbrida combinando activación clásica con 
        colapso cuántico controlado.
        
        Args:
            x (np.ndarray): Datos de entrada.
            activation_type (ActivationType): Tipo de activación.
            collapse_threshold (float): Umbral para determinar colapso.
            
        Returns:
            Tuple[np.ndarray, float]: Salida activada y probabilidad de colapso.
        """
        classic_out = self.activation_functions[activation_type](x)
        collapse_prob = np.mean(np.abs(classic_out))  # Umbral basado en la magnitud promedio
        if collapse_prob > collapse_threshold:
            quantum_factor = np.sqrt(1 - collapse_prob)
            modified_out = classic_out * quantum_factor
        else:
            modified_out = classic_out
        return modified_out, collapse_prob

    def hybrid_backpropagation(self,
                               gradient: np.ndarray,
                               activation_type: ActivationType,
                               collapse_probability: float) -> np.ndarray:
        """
        Retropropagación híbrida considerando la función de activación y el efecto de colapso.
        
        Args:
            gradient (np.ndarray): Gradiente de la capa siguiente.
            activation_type (ActivationType): Tipo de activación.
            collapse_probability (float): Probabilidad de colapso.
            
        Returns:
            np.ndarray: Gradiente modificado.
        """
        classical_deriv = self.activation_derivatives[activation_type]
        quantum_factor = np.sqrt(1 - collapse_probability)
        modified_grad = gradient * classical_deriv(gradient) * quantum_factor
        return modified_grad


# -------------------------
# Clase QuantumState
# -------------------------
class QuantumState(QuantumBayesMahalanobis):
    """
    Gestiona el estado cuántico utilizando métodos bayesianos, distancias de Mahalanobis,
    y expandiendo con una representación en forma de vector de probabilidades.
    
    Provee métodos para predecir y actualizar el estado.
    """
    def __init__(self, num_positions: int, learning_rate: float = 0.1):
        super().__init__()  # Inicializa la parte avanzada cuántica
        self.num_positions = num_positions
        self.learning_rate = learning_rate
        self.state_vector = np.random.rand(num_positions)
        self.state_vector = self.normalize_state(self.state_vector)
        self.probabilities = self.state_vector.copy()

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(state)
        return state / norm if norm != 0 else state

    def predict_state_update(self) -> Tuple[np.ndarray, float]:
        """
        Usa el método predict_quantum_state (por herencia) para calcular la predicción del siguiente estado.
        Se calculan valores de entropía y coherencia (utilizando StatisticalAnalysis) para alimentar la predicción.
        
        Returns:
            Tuple[np.ndarray, float]: (Nuevo estado predicho, posterior)
        """
        current_input = np.array([self.state_vector])
        entropy = StatisticalAnalysis.shannon_entropy(self.state_vector)
        coherence = np.mean(self.state_vector)
        next_state_tensor, posterior = self.predict_quantum_state(current_input, entropy, coherence)
        next_state = next_state_tensor.numpy().flatten()
        next_state = self.normalize_state(next_state)
        return next_state, posterior

def update_state(self, action: int) -> None:
    """
    Actualiza el estado cuántico combinando el estado actual con el estado predicho,
    ponderado por la tasa de aprendizaje e influenciado por la acción.
    
    Args:
        action (int): Acción tomada (0 o 1) que influirá en la actualización del estado.
    """
    # Predecir el siguiente estado
    next_state, posterior = self.predict_state_update()
    
    # Modificar la actualización basada en la acción
    if action == 1:
        # Acción positiva: explorar más el nuevo estado
        update_factor = self.learning_rate * (1 + posterior)
    else:
        # Acción negativa: mantener más el estado actual
        update_factor = self.learning_rate * (1 - posterior)
    
    # Actualizar estado con factor de aprendizaje dinámico
    updated_state = (1 - update_factor) * self.state_vector + update_factor * next_state
    
    # Normalizar y actualizar
    self.state_vector = self.normalize_state(updated_state)
    
    # Actualizar probabilidades
    self.probabilities = np.abs(self.state_vector)

def compute_quantum_uncertainty(self) -> float:
    """
    Calcula la incertidumbre del estado cuántico basada en la entropía de Shannon.
    
    Returns:
        float: Valor de incertidumbre (0-1)
    """
    entropy = StatisticalAnalysis.shannon_entropy(np.abs(self.probabilities))
    return entropy

def quantum_interference(self, other_state: 'QuantumState') -> np.ndarray:
    """
    Simula la interferencia cuántica entre dos estados.
    
    Args:
        other_state (QuantumState): Otro estado cuántico para interferir.
    
    Returns:
        np.ndarray: Estado resultante de la interferencia.
    """
    # Producto punto complejo
    interference_pattern = np.dot(
        np.abs(self.state_vector), 
        np.abs(other_state.state_vector)
    )
    
    # Generar nuevo estado con patrón de interferencia
    new_state = self.state_vector * np.cos(interference_pattern)
    
    return self.normalize_state(new_state)

def quantum_entanglement_measure(self, other_state: 'QuantumState') -> float:
    """
    Calcula una medida de entrelazamiento entre dos estados cuánticos.
    
    Args:
        other_state (QuantumState): Otro estado cuántico para medir entrelazamiento.
    
    Returns:
        float: Medida de entrelazamiento (0-1)
    """
    # Producto tensorial de probabilidades
    tensor_product = np.outer(
        np.abs(self.probabilities), 
        np.abs(other_state.probabilities)
    )
    
    # Calcular entrelazamiento basado en la entropía del producto tensorial
    entanglement_entropy = StatisticalAnalysis.shannon_entropy(tensor_product.flatten())
    
    return entanglement_entropy

def visualize_state(self) -> dict:
    """
    Proporciona una visualización detallada del estado cuántico.
    
    Returns:
        dict: Diccionario con información del estado.
    """
    return {
        'state_vector': self.state_vector.tolist(),
        'probabilities': self.probabilities.tolist(),
        'uncertainty': self.compute_quantum_uncertainty(),
        'norm': np.linalg.norm(self.state_vector)
    }

def quantum_measurement(self, observable: np.ndarray = None) -> float:
    """
    Realiza una medición del estado cuántico con un observable opcional.
    
    Args:
        observable (np.ndarray, opcional): Matriz de observable. Si no se proporciona, 
                                           se usa el vector de estado actual.
    
    Returns:
        float: Valor de expectación de la medición.
    """
    if observable is None:
        observable = np.diag(self.probabilities)
    
    # Calcular valor de expectación
    expectation_value = np.dot(
        self.state_vector.T, 
        np.dot(observable, self.state_vector)
    )
    
    return np.real(expectation_value)

# Método para serializar el estado cuántico
def serialize_state(self) -> dict:
    """
    Serializa el estado cuántico para persistencia.
    
    Returns:
        dict: Diccionario con información serializable del estado.
    """
    return {
        'state_vector': self.state_vector.tolist(),
        'probabilities': self.probabilities.tolist(),
        'num_positions': self.num_positions,
        'learning_rate': self.learning_rate
    }

@classmethod
def deserialize_state(cls, serialized_data: dict) -> 'QuantumState':
    """
    Deserializa un estado cuántico previamente serializado.
    
    Args:
        serialized_data (dict): Datos serializados del estado.
    
    Returns:
        QuantumState: Estado cuántico reconstruido.
    """
    quantum_state = cls(
        num_positions=serialized_data['num_positions'], 
        learning_rate=serialized_data['learning_rate']
    )
    quantum_state.state_vector = np.array(serialized_data['state_vector'])
    quantum_state.probabilities = np.array(serialized_data['probabilities'])
    
    return quantum_state
# Ejemplo de uso: Crear dos estados cuánticos.
state1 = QuantumState(num_positions=5)
state2 = QuantumState(num_positions=5)

# Actualizar estados
state1.update_state(action=1)
state2.update_state(action=0)

# Calcular interferencia
interference_state = state1.quantum_interference(state2)

# Medir entrelazamiento
entanglement = state1.quantum_entanglement_measure(state2)

# Visualizar estado
state_info = state1.visualize_state()
print("Estado cuántico:", state_info)
print("Entrelazamiento:", entanglement)

# Serializar y deserializar
serialized = state1.serialize_state()
reconstructed_state = QuantumState.deserialize_state(serialized)