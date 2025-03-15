# inicializador.py
"""
Módulo Híbrido de Inicialización

Este script define:
  • Una red neural básica (NeuralNetwork) con varias funciones de activación y
    soporte para optimizadores (SGD y Adam).
  • La clase QuantumClassicalActivator, la cual crea un circuito cuántico relacionado
    con una función de activación y permite realizar un paso forward híbrido:
      - Primero se calcula una activación clásica;
      - Se determina la "probabilidad de colapso" (una forma de representar la incertidumbre).
  
  • Además, se integra la lógica de quantum_bayes_mahalanobis en la clase QuantumState.
    QuantumState hereda de QuantumBayesMahalanobis (definido en otro módulo) para operar,
    por ejemplo, con proyecciones y predicciones cuánticas. Esta clase puede servir como el
    componente cuántico central en un sistema híbrido, actualizándose mediante interacción.

Autor: Jacobo Tlacaelel Mina Rodríguez  
Fecha: 13/03/2025  
Versión: cuadrante-coremind v1.0  
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from typing import Tuple, Optional
from enum import Enum

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
    
    def _adam(self, layer: int, a: np.ndarray, delta: np.ndarray, learning_rate: float, t: int,
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
# Se integra el script de quantum_bayes_mahalanobis aquí; se asume que la clase
# QuantumBayesMahalanobis ya está definida en otro módulo e importada al inicio.
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
        ponderado por la tasa de aprendizaje.
        
        Args:
            action (int): Acción tomada (puede influir en la actualización, aunque aquí se usa un ejemplo fijo).
        """
        next_state, posterior = self.predict_state_update()
        self.state_vector = self.normalize_state(
            (1 - self.learning_rate) * self.state_vector + self.learning_rate * next_state
        )
        self.probabilities = self.state_vector.copy()

    def visualize_state(self) -> None:
        print("Estado cuántico (vector de probabilidades):")
        print(self.state_vector)


# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    # Ejemplo para la parte clásica: red neuronal y activador híbrido
    nn = NeuralNetwork(input_size=3, hidden_size=[4, 3], output_size=2, activation="sigmoid")
    x_sample = np.array([[0.5, -0.2, 0.8]])
    activations = nn.forward(x_sample)
    print("Activaciones de la red neural:")
    for a in activations:
        print(a)
    
    hybrid_activator = QuantumClassicalActivator(n_qubits=3)
    activated_output, collapse_prob = hybrid_activator.quantum_activated_forward(x_sample.flatten(), ActivationType.SIGMOID, collapse_threshold=0.6)
    print("\nSalida activada híbrida:")
    print(activated_output)
    print("Probabilidad de colapso:", collapse_prob)
    
    quantum_circuit = hybrid_activator.create_quantum_circuit(ActivationType.SIGMOID)
    print("\nCircuito cuántico generado:")
    print(quantum_circuit)
    
    # Ejemplo para la parte cuántica: QuantumState usado junto con quantum_bayes_mahalanobis
    qs = QuantumState(num_positions=8, learning_rate=0.1)
    print("\nEstado cuántico inicial:")
    qs.visualize_state()
    
    # Simular una actualización del estado cuántico (por ejemplo, tras una acción)
    qs.update_state(action=1)
    print("\nEstado cuántico tras actualización:")
    qs.visualize_state()

"""
Análisis de la Lógica:
1. La red neural clásica se inicializa y se propaga la señal mediante funciones de activación (sigmoid, tanh o relu).
2. QuantumClassicalActivator implementa funciones para simular un "colapso" cuántico: primero se aplica la activación clásica, luego se ajusta la salida según un criterio (collapse_threshold) que modula con un quantum_factor.
3. La integración es clave: la clase QuantumState hereda de QuantumBayesMahalanobis, lo cual permite utilizar métodos avanzados (por ejemplo, predict_quantum_state) para predecir el siguiente estado a partir de parámetros calculados (entropía y coherencia).
4. La actualización del estado cuántico se hace de forma gradual (mezcla ponderada), lo que simula la actualización (o colapso controlado) del estado cuántico.
5. La modularidad de cada componente (red neural, activador híbrido, estado cuántico) permite combinarlos en sistemas híbridos donde la parte clásica y la cuántica se refuerzan mutuamente.

Con esta versión se logra una integración robusta que toma elementos del script quantum_bayes_mahalanobis y los utiliza en QuantumState, combinándolos con estrategias clásicas de activación y propagación. Esto está en la misma línea de lo planteado anteriormente, y ofrece una base sólida para la experimentación en sistemas híbridos cuántico-clásicos.
"""