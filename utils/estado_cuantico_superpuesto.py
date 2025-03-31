#!/usr/bin/env python3

""" lógica del: 
estado_superpuesto = {
    "00": (0.707 + 0j),  # Amplitud compleja del estado |00⟩ (ejemplo)
    "01": (0.0 + 0.0j),
    "10": (0.0 + 0.0j),
    "11": (0.707 + 0j)   # Amplitud compleja del estado |11⟩ (ejemplo)
}

variable_estado = estado_superpuesto

Archivo Unificado: Sistema Híbrido Cuántico-Bayesiano con Representación de Momentum en Superposición

Este archivo combina:
1. `QuantumBayesianHybridSystem`: Sistema híbrido que integra RNNs, análisis de Mahalanobis y lógica bayesiana.
2. Circuitos cuánticos resistentes para generar datos reales como entrada.
3. Representación del momentum cuántico en estados de superposición.

Autor: Jacobo Tlacaelel Mina Rodríguez
Fecha: 31/03/2025
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from qiskit import Aer, execute, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Importar clases del sistema híbrido y del módulo de circuitos cuánticos.
from quantum_hybrid_system import QuantumBayesianHybridSystem
from quantum_fourier_network import ResilientQuantumCircuit

class QuantumMomentumRepresentation:
    """
    Clase para representar y manipular el momentum cuántico en estados de superposición.
    """
    def __init__(self, num_qubits=2):
        """
        Inicializa la representación del momentum cuántico.
        
        Args:
            num_qubits (int): Número de qubits en el sistema.
        """
        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        self.momentum_space = None
        self.position_space = None
        
    def load_superposition_state(self, state_dict):
        """
        Carga un estado de superposición desde un diccionario.
        
        Args:
            state_dict (dict): Diccionario con las amplitudes complejas del estado.
                               Las claves son strings de bits ("00", "01", etc.)
                               Los valores son números complejos representando las amplitudes.
        """
        # Crear un vector de estado vacío
        state_vector = np.zeros(self.dimension, dtype=complex)
        
        # Llenar el vector con las amplitudes proporcionadas
        for basis_state, amplitude in state_dict.items():
            # Convertir la string binaria a un índice decimal
            index = int(basis_state, 2)
            state_vector[index] = amplitude
            
        # Normalizar el vector (por seguridad)
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        self.position_space = state_vector
        
        # Calcular la representación en el espacio de momentum usando la transformada de Fourier
        self.calculate_momentum_space()
        
        return state_vector
    
    def calculate_momentum_space(self):
        """
        Calcula la representación del estado en el espacio de momentum
        utilizando la transformada cuántica de Fourier.
        """
        if self.position_space is None:
            raise ValueError("Debe cargar un estado en el espacio de posición primero")
        
        # Usar la Transformada de Fourier para obtener la representación de momentum
        self.momentum_space = fft(self.position_space) / np.sqrt(self.dimension)
        
        return self.momentum_space
    
    def calculate_position_space(self):
        """
        Calcula la representación del estado en el espacio de posición
        utilizando la transformada inversa de Fourier.
        """
        if self.momentum_space is None:
            raise ValueError("Debe calcular el espacio de momentum primero")
        
        # Usar la Transformada Inversa de Fourier para obtener la representación de posición
        self.position_space = ifft(self.momentum_space) * np.sqrt(self.dimension)
        
        return self.position_space
    
    def get_momentum_probabilities(self):
        """
        Obtiene las probabilidades de medir cada valor de momentum.
        
        Returns:
            np.ndarray: Vector de probabilidades para cada valor de momentum.
        """
        if self.momentum_space is None:
            self.calculate_momentum_space()
            
        return np.abs(self.momentum_space)**2
    
    def get_position_probabilities(self):
        """
        Obtiene las probabilidades de medir cada valor de posición.
        
        Returns:
            np.ndarray: Vector de probabilidades para cada valor de posición.
        """
        return np.abs(self.position_space)**2
    
    def plot_momentum_distribution(self):
        """
        Genera una gráfica de la distribución de probabilidad del momentum.
        """
        probs = self.get_momentum_probabilities()
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.dimension), probs)
        plt.xlabel('Valor de momentum')
        plt.ylabel('Probabilidad')
        plt.title('Distribución de probabilidad del momentum cuántico')
        plt.xticks(range(self.dimension), [bin(i)[2:].zfill(self.num_qubits) for i in range(self.dimension)])
        plt.grid(alpha=0.3)
        return plt
    
    def create_qiskit_circuit(self):
        """
        Crea un circuito cuántico de Qiskit que representa el estado actual.
        
        Returns:
            QuantumCircuit: Circuito cuántico que representa el estado.
        """
        circuit = QuantumCircuit(self.num_qubits)
        
        # Inicializar el circuito con el estado en el espacio de posición
        circuit.initialize(self.position_space, range(self.num_qubits))
        
        return circuit
    
    def visualize_bloch_sphere(self):
        """
        Visualiza el estado cuántico en la esfera de Bloch (solo para 1 o 2 qubits).
        """
        if self.num_qubits > 2:
            raise ValueError("La visualización en la esfera de Bloch solo es útil para 1 o 2 qubits")
        
        state = Statevector(self.position_space)
        fig = plot_bloch_multivector(state)
        return fig
    
    def quantum_fourier_transform_circuit(self):
        """
        Crea un circuito cuántico que aplica la Transformada Cuántica de Fourier.
        
        Returns:
            QuantumCircuit: Circuito con la QFT aplicada.
        """
        circuit = self.create_qiskit_circuit()
        
        # Aplicar QFT
        for i in range(self.num_qubits):
            circuit.h(i)
            for j in range(i + 1, self.num_qubits):
                circuit.cp(2 * np.pi / (2 ** (j - i)), j, i)
                
        # Invertir el orden de los qubits (necesario para la QFT estándar)
        for i in range(self.num_qubits // 2):
            circuit.swap(i, self.num_qubits - i - 1)
            
        return circuit

def generate_quantum_data(num_qubits: int = 5, superposition_state=None) -> np.ndarray:
    """
    Genera estados cuánticos utilizando un circuito resistente o un estado de superposición dado.
    
    Args:
        num_qubits (int): Número de qubits en el circuito.
        superposition_state (dict, optional): Estado de superposición definido como diccionario.
        
    Returns:
        np.ndarray: Amplitudes complejas del estado cuántico generado.
    """
    if superposition_state:
        # Usar el estado de superposición proporcionado
        qmr = QuantumMomentumRepresentation(num_qubits=len(list(superposition_state.keys())[0]))
        state_vector = qmr.load_superposition_state(superposition_state)
        
        # Obtener también la representación en el espacio de momentum
        momentum_vector = qmr.momentum_space
        
        # Convertir a matriz NumPy (componentes real e imaginaria separadas)
        quantum_states = np.array([
            state_vector.real, 
            state_vector.imag,
            momentum_vector.real,
            momentum_vector.imag
        ]).T
    else:
        # Crear un circuito resistente
        circuit = ResilientQuantumCircuit(num_qubits)
        circuit.create_resilient_state()

        # Obtener las amplitudes complejas del estado cuántico
        state_vector = circuit.get_complex_amplitudes()

        # Convertir a matriz NumPy (real e imaginario separados)
        quantum_states = np.array([state_vector.real, state_vector.imag]).T
    
    return quantum_states

def quantum_hybrid_simulation(estado_superpuesto=None):
    """
    Simulación del sistema híbrido utilizando datos reales de circuitos cuánticos
    y representando el momentum cuántico en superposición.
    
    Args:
        estado_superpuesto (dict, optional): Diccionario con el estado de superposición.
    """
    # Configuración inicial del sistema híbrido
    input_size = 5
    hidden_size = 64
    output_size = 2

    # Crear instancia del sistema híbrido
    quantum_hybrid_system = QuantumBayesianHybridSystem(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        prn_influence=0.5
    )

    # Determinar el número de qubits a partir del estado superpuesto o usar el predeterminado
    if estado_superpuesto:
        num_qubits = len(list(estado_superpuesto.keys())[0])
    else:
        num_qubits = input_size
    
    # Generar datos reales desde un circuito cuántico o estado superpuesto
    quantum_states = generate_quantum_data(num_qubits=num_qubits, superposition_state=estado_superpuesto)

    # Crear una instancia de la representación de momentum cuántico
    qmr = QuantumMomentumRepresentation(num_qubits=num_qubits)
    
    if estado_superpuesto:
        # Cargar el estado superpuesto
        qmr.load_superposition_state(estado_superpuesto)
        
        # Visualizar la distribución de momentum
        momentum_fig = qmr.plot_momentum_distribution()
        momentum_fig.savefig("momentum_distribution.png")
        
        # Si es posible, visualizar en la esfera de Bloch
        if num_qubits <= 2:
            try:
                bloch_fig = qmr.visualize_bloch_sphere()
                bloch_fig.savefig("bloch_sphere.png")
            except Exception as e:
                print(f"No se pudo visualizar la esfera de Bloch: {e}")
    
    # Calcular entropía y coherencia a partir de los estados cuánticos generados
    entropy = quantum_hybrid_system.statistical_analyzer.shannon_entropy(quantum_states.flatten())
    
    # La coherencia se calcula como la suma de los elementos fuera de la diagonal de la matriz de densidad
    if estado_superpuesto:
        # Calcular la matriz de densidad para el estado superpuesto
        state_vector = qmr.position_space
        density_matrix = np.outer(state_vector, np.conjugate(state_vector))
        
        # Calculamos la coherencia como la suma de los elementos fuera de la diagonal
        coherence = np.sum(np.abs(density_matrix - np.diag(np.diag(density_matrix))))
    else:
        coherence = np.mean(quantum_states)

    # Entrenar el sistema híbrido con los datos generados
    quantum_hybrid_system.train_hybrid_system(
        quantum_states,
        entropy,
        coherence,
        epochs=50
    )

    # Predecir el siguiente estado utilizando el sistema entrenado
    prediction = quantum_hybrid_system.predict_quantum_state(
        quantum_states[-1:],
        entropy,
        coherence
    )

    # Optimizar un estado objetivo
    # Para el caso de superposición, buscamos un estado que maximice el momentum
    if estado_superpuesto:
        # El target es un estado con alta probabilidad en los valores altos de momentum
        target_state = np.zeros(2**num_qubits)
        target_state[-1] = 1.0  # Máxima probabilidad en el momentum más alto
    else:
        target_state = np.array([1.0, 0.0, 0.5, 0.5, 0.0])
    
    optimized_states, objective = quantum_hybrid_system.optimize_quantum_state(
        quantum_states,
        target_state
    )

    # Imprimir resultados de la simulación
    print("\n===== RESULTADOS DE LA SIMULACIÓN =====")
    print("\nEstados de superposición y momentum:")
    
    if estado_superpuesto:
        print("\nEstado de superposición proporcionado:")
        for basis, amplitude in estado_superpuesto.items():
            prob = np.abs(amplitude)**2
            print(f"|{basis}⟩: Amplitud = {amplitude}, Probabilidad = {prob:.4f}")
        
        print("\nRepresentación en el espacio de momentum:")
        momentum_probs = qmr.get_momentum_probabilities()
        momentum_amplitudes = qmr.momentum_space
        for i, (amp, prob) in enumerate(zip(momentum_amplitudes, momentum_probs)):
            basis = bin(i)[2:].zfill(num_qubits)
            print(f"|{basis}⟩: Amplitud = {amp}, Probabilidad = {prob:.4f}")
    
    print("\nPredecciones del sistema híbrido:")
    print("Predicción RNN:", prediction['rnn_prediction'])
    print("Predicción Bayesiana:", prediction['bayes_prediction'])
    print("Predicción Combinada:", prediction['combined_prediction'])
    
    print("\nDatos de optimización:")
    print("Estados Optimizados:", optimized_states)
    print("Objetivo de Optimización:", objective)
    
    print("\nPropiedades cuánticas:")
    print("Entropía de Shannon:", entropy)
    print("Coherencia cuántica:", coherence)
    
    # Si se generaron visualizaciones, informar al usuario
    if estado_superpuesto:
        print("\nSe han generado visualizaciones:")
        print("- momentum_distribution.png: Distribución de probabilidad del momentum")
        if num_qubits <= 2:
            print("- bloch_sphere.png: Representación en la esfera de Bloch")

if __name__ == "__main__":
    # Estado de superposición (ejemplo: estado de Bell |Φ+⟩ = (|00⟩ + |11⟩)/√2)
    estado_superpuesto = {
        "00": (0.707 + 0j),  # Amplitud compleja del estado |00⟩ (ejemplo)
        "01": (0.0 + 0.0j),
        "10": (0.0 + 0.0j),
        "11": (0.707 + 0j)   # Amplitud compleja del estado |11⟩ (ejemplo)
    }
    
    variable_estado = estado_superpuesto
    
    # Ejecutar la simulación con el estado superpuesto
    quantum_hybrid_simulation(variable_estado)