#!/usr/bin/env python3
"""
Módulo: IntegratedQuantumNetwork

Este módulo integra circuitos cuánticos resistentes con análisis FFT y bayesiano.
Combina la creación de estados cuánticos mediante Qiskit con procesamiento
avanzado basado en FFT para generar features e inicializar redes neuronales.

Autor: Jacobo Tlacaelel Mina Rodríguez (optimizado y documentación por Gemini, Claude y ChatGPT)
Fecha: 16/03/2025
Versión: cuadrante-coremind v1.2
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.covariance import EmpiricalCovariance
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PRN:
    """
    Clase para modelar el Ruido Probabilístico de Referencia (Probabilistic Reference Noise).

    Esta clase generalizada puede ser utilizada para representar cualquier tipo de
    influencia probabilística en un sistema.

    Atributos:
        influence (float): Factor de influencia entre 0 y 1.
        algorithm_type (str, opcional): Tipo de algoritmo a utilizar.
        parameters (dict, opcional): Parámetros adicionales específicos del algoritmo.
    """
    def __init__(self, influence: float, algorithm_type: Optional[str] = None, **parameters: dict):
        """
        Inicializa un objeto PRN con un factor de influencia y parámetros específicos.

        Args:
            influence (float): Factor de influencia entre 0 y 1.
            algorithm_type (str, opcional): Tipo de algoritmo a utilizar.
            **parameters: Parámetros adicionales específicos del algoritmo.

        Raises:
            ValueError: Si influence está fuera del rango [0,1].
        """
        if not 0 <= influence <= 1:
            raise ValueError(f"La influencia debe estar entre 0 y 1. Valor recibido: {influence}")

        self.influence = influence
        self.algorithm_type = algorithm_type
        self.parameters = parameters

    def adjust_influence(self, adjustment: float) -> None:
        """
        Ajusta el factor de influencia dentro de los límites permitidos.

        Args:
            adjustment (float): Valor de ajuste (positivo o negativo).

        Raises:
            ValueError: Si el nuevo valor de influencia está fuera del rango [0,1].
        """
        new_influence = self.influence + adjustment

        if not 0 <= new_influence <= 1:
            # Truncamos al rango válido
            new_influence = max(0, min(1, new_influence))
            print(f"ADVERTENCIA: Influencia ajustada a {new_influence} para mantenerla en el rango [0,1]")

        self.influence = new_influence

    def combine_with(self, other_prn: 'PRN', weight: float = 0.5) -> 'PRN':
        """
        Combina este PRN con otro según un peso específico.

        Args:
            other_prn (PRN): Otro objeto PRN para combinar.
            weight (float): Peso para la combinación, entre 0 y 1 (por defecto 0.5).

        Returns:
            PRN: Un nuevo objeto PRN con la influencia combinada.

        Raises:
            ValueError: Si weight está fuera del rango [0,1].
        """
        if not 0 <= weight <= 1:
            raise ValueError(f"El peso debe estar entre 0 y 1. Valor recibido: {weight}")

        # Combinación ponderada de las influencias
        combined_influence = self.influence * weight + other_prn.influence * (1 - weight)

        # Combinar los parámetros de ambos PRN
        combined_params = {**self.parameters, **other_prn.parameters}

        # Elegir el tipo de algoritmo según el peso
        algorithm = self.algorithm_type if weight >= 0.5 else other_prn.algorithm_type

        return PRN(combined_influence, algorithm, **combined_params)

    def record_noise(self, probabilities: Dict[str, float]) -> float:
        """
        Registra ruido basado en las probabilidades y calcula la entropía.

        Args:
            probabilities (dict): Diccionario de probabilidades.

        Returns:
            float: Entropía calculada.
        """
        values = list(probabilities.values())
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in values)
        return entropy

    def __str__(self) -> str:
        """
        Representación en string del objeto PRN.

        Returns:
            str: Descripción del objeto PRN.
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"PRN(influence={self.influence:.4f}{algo_str}{', ' + params_str if params_str else ''})"


class ComplexPRN(PRN):
    """
    Clase PRN modificada para representar números complejos.

    Hereda de la clase PRN y utiliza el módulo del número complejo como factor de influencia.

    Atributos:
        real_component (float): Componente real del número complejo.
        imaginary_component (float): Componente imaginaria del número complejo.
    """
    def __init__(self, real_component: float, imaginary_component: float, algorithm_type: Optional[str] = None, **parameters: dict):
        """
        Inicializa un PRN complejo.

        Args:
            real_component (float): Componente real.
            imaginary_component (float): Componente imaginaria.
            algorithm_type (str, opcional): Tipo de algoritmo a utilizar.
            **parameters: Parámetros específicos adicionales.
        """
        self.real_component = real_component
        self.imaginary_component = imaginary_component
        # Cálculo del módulo como factor de influencia
        influence = np.sqrt(real_component**2 + imaginary_component**2)
        # Normalizar la influencia a [0,1]
        normalized_influence = min(1.0, influence)

        super().__init__(normalized_influence, algorithm_type, **parameters)

    def get_phase(self) -> float:
        """
        Calcula la fase del número complejo.

        Returns:
            float: Fase en radianes.
        """
        return np.arctan2(self.imaginary_component, self.real_component)

    def __str__(self) -> str:
        """
        Representación en string del objeto ComplexPRN.

        Returns:
            str: Descripción del objeto ComplexPRN.
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"ComplexPRN(real={self.real_component:.4f}, imag={self.imaginary_component:.4f}, influence={self.influence:.4f}{algo_str}{', ' + params_str if params_str else ''})"


class EnhancedPRN(PRN):
    """
    Extiende la clase PRN para registrar distancias de Mahalanobis y con ello
    definir un 'ruido cuántico' adicional en el sistema.

    Atributos:
        mahalanobis_records (list): Lista para almacenar los valores promedio de la distancia de Mahalanobis registrados.
    """
    def __init__(self, influence: float = 0.5, algorithm_type: Optional[str] = None, **parameters: dict):
        """
        Constructor que permite definir la influencia y el tipo de algoritmo,
        además de inicializar una lista para conservar registros promedio de
        distancias de Mahalanobis.

        Args:
            influence (float, opcional): Factor de influencia entre 0 y 1. Por defecto 0.5.
            algorithm_type (str, opcional): Tipo de algoritmo a utilizar.
            **parameters: Parámetros adicionales específicos del algoritmo.
        """
        super().__init__(influence, algorithm_type, **parameters)
        self.mahalanobis_records = []

    def record_quantum_noise(self, probabilities: Dict[str, float], quantum_states: np.ndarray) -> Tuple[float, float]:
        """
        Registra un 'ruido cuántico' basado en la distancia de Mahalanobis
        calculada para los estados cuánticos.

        Args:
            probabilities (dict): Diccionario de probabilidades (ej. {"0": p_0, "1": p_1, ...}).
            quantum_states (numpy.ndarray): Estados cuánticos (n_muestras, n_dimensiones).

        Returns:
            Tuple[float, float]: Una tupla que contiene:
                - entropy (float): Entropía calculada a partir de probabilities.
                - mahal_mean (float): Distancia promedio de Mahalanobis.
        """
        # Calculamos la entropía
        entropy = self.record_noise(probabilities)

        # Ajuste del estimador de covarianza
        cov_estimator = EmpiricalCovariance().fit(quantum_states)
        mean_state = np.mean(quantum_states, axis=0)
        inv_cov = np.linalg.pinv(cov_estimator.covariance_)

        # Cálculo vectorizado de la distancia
        diff = quantum_states - mean_state
        aux = diff @ inv_cov
        dist_sqr = np.einsum('ij,ij->i', aux, diff)
        distances = np.sqrt(dist_sqr)
        mahal_mean = np.mean(distances)

        # Se registra la distancia promedio
        self.mahalanobis_records.append(mahal_mean)

        return entropy, mahal_mean


class StatisticalAnalysis:
    """
    Clase para realizar análisis estadísticos comunes en el contexto del ruido probabilístico.
    """
    def shannon_entropy(self, data: Union[list, np.ndarray]) -> float:
        """
        Calcula la entropía de Shannon de un conjunto de datos.

        Args:
            data (list o numpy.ndarray): Lista o array de datos.

        Returns:
            float: Entropía de Shannon en bits.
        """
        # 1. Contar ocurrencias de cada valor único:
        values, counts = np.unique(data, return_counts=True)

        # 2. Calcular probabilidades:
        probabilities = counts / len(data)

        # 3. Evitar logaritmos de cero:
        probabilities = probabilities[probabilities > 0]

        # 4. Calcular la entropía:
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def calculate_cosines(self, entropy: float, env_value: float) -> Tuple[float, float, float]:
        """
        Calcula los cosenos direccionales (x, y, z) para un vector tridimensional.

        Args:
            entropy (float): Entropía de Shannon (componente x).
            env_value (float): Valor contextual del entorno (componente y).

        Returns:
            tuple: Cosenos direccionales (cos_x, cos_y, cos_z).
        """
        # Asegurar evitar división por cero:
        if entropy == 0:
            entropy = 1e-6
        if env_value == 0:
            env_value = 1e-6

        # Magnitud del vector tridimensional:
        magnitude = np.sqrt(entropy ** 2 + env_value ** 2 + 1)

        # Cálculo de cosenos direccionales:
        cos_x = entropy / magnitude
        cos_y = env_value / magnitude
        cos_z = 1 / magnitude

        return cos_x, cos_y, cos_z


class BayesLogic:
    """
    Implementa la lógica bayesiana para actualizar creencias basadas en evidencia.

    Atributos:
        prior (Dict[str, float]): Distribución previa de probabilidades.
        posterior (Dict[str, float]): Distribución posterior de probabilidades.
    """
    def __init__(self, prior: Optional[Dict[str, float]] = None):
        """
        Inicializa la lógica bayesiana con creencias previas opcionales.

        Args:
            prior (Dict[str, float], opcional): Distribución previa de probabilidades.
        """
        self.prior = prior or {}
        self.posterior = {}

    def update(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """
        Actualiza las creencias posteriores basadas en nueva evidencia.

        Args:
            evidence (Dict[str, float]): Nueva evidencia como probabilidades.

        Returns:
            Dict[str, float]: Distribución posterior actualizada.
        """
        # Si no hay prior, usar la evidencia como prior
        if not self.prior:
            self.prior = {k: 1/len(evidence) for k in evidence.keys()}

        # Calcular normalización
        total = sum(self.prior[k] * evidence.get(k, 1) for k in self.prior)

        # Actualizar posterior
        self.posterior = {k: (self.prior[k] * evidence.get(k, 1)) / total for k in self.prior}

        return self.posterior

    def get_maximum_posterior(self) -> Tuple[Optional[str], float]:
        """
        Obtiene la hipótesis con mayor probabilidad posterior.

        Returns:
            Tuple[str, float]: Tupla con (hipótesis, probabilidad). Retorna (None, 0) si no hay posterior calculada.
        """
        if not self.posterior:
            return None, 0

        max_key = max(self.posterior, key=self.posterior.get)
        return max_key, self.posterior[max_key]


class ResilientQuantumCircuit:
    """
    Simula un circuito cuántico resistente a errores.

    Atributos:
        n_qubits (int): Número de qubits en el circuito.
        n_states (int): Número de estados posibles (2^n_qubits).
        state_vector (numpy.ndarray): Vector de estado del circuito cuántico.
    """
    def __init__(self, n_qubits: int):
        """
        Inicializa un circuito cuántico con n qubits.

        Args:
            n_qubits (int): Número de qubits en el circuito.
        """
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        # Inicializar en estado |0...0⟩
        self.state_vector = np.zeros(self.n_states, dtype=complex)
        self.state_vector[0] = 1.0

    def apply_gate(self, gate: np.ndarray, target_qubits: List[int]) -> None:
        """
        Aplica una compuerta cuántica a los qubits objetivo.

        Args:
            gate (numpy.ndarray): Matriz de la compuerta cuántica.
            target_qubits (List[int]): Lista de qubits objetivo.

        Nota:
            Esta es una implementación simplificada. En un caso real, se usarían
            operaciones tensoriales para aplicar la compuerta al subespacio correcto.
        """
        # Implementación simplificada - en un caso real se usaría
        # operaciones tensoriales para aplicar la compuerta
        pass

    def get_complex_amplitudes(self) -> List[complex]:
        """
        Obtiene las amplitudes complejas del estado actual.

        Returns:
            List[complex]: Lista de amplitudes complejas.
        """
        return self.state_vector.tolist()

    def get_probabilities(self) -> Dict[str, float]:
        """
        Calcula las probabilidades de cada estado.

        Returns:
            Dict[str, float]: Diccionario de probabilidades por estado, donde las claves son las representaciones binarias de los estados.
        """
        probs = np.abs(self.state_vector)**2
        return {format(i, f"0{self.n_qubits}b"): prob for i, prob in enumerate(probs)}


class FFTBayesIntegrator:
    """
    Clase que integra la Transformada Rápida de Fourier (FFT) con el análisis bayesiano
    para procesar señales cuánticas y generar representaciones para la inicialización informada
    de modelos o como features para redes neuronales.

    Atributos:
        bayes_logic (BayesLogic): Instancia de la clase BayesLogic para realizar inferencia bayesiana.
        stat_analysis (StatisticalAnalysis): Instancia de la clase StatisticalAnalysis para realizar cálculos estadísticos.
        cache (Dict[int, Dict[str, Union[numpy.ndarray, float]]]): Caché para almacenar resultados del procesamiento de estados cuánticos.
    """
    def __init__(self) -> None:
        """
        Inicializa instancias de lógica bayesiana y análisis estadístico, además de una caché.
        """
        self.bayes_logic = BayesLogic()
        self.stat_analysis = StatisticalAnalysis()
        self.cache: Dict[int, Dict[str, Union[np.ndarray, float]]] = {}  # Caché para resultados calculados.

    def process_quantum_circuit(self, quantum_circuit: ResilientQuantumCircuit) -> Dict[str, Union[np.ndarray, float]]:
        """
        Procesa un circuito cuántico resistente aplicando la FFT a su estado.

        Args:
            quantum_circuit (ResilientQuantumCircuit): Circuito cuántico a procesar.

        Returns:
            Dict[str, Union[numpy.ndarray, float]]: Un diccionario que contiene las magnitudes y fases de la FFT, la entropía y la coherencia.
        """
        amplitudes = quantum_circuit.get_complex_amplitudes()  # Obtiene las amplitudes complejas del estado.
        return self.process_quantum_state(amplitudes)  # Procesa las amplitudes usando FFT.

    def process_quantum_state(self, quantum_state: List[complex]) -> Dict[str, Union[np.ndarray, float]]:
        """
        Procesa un estado cuántico aplicando la FFT y extrayendo características frecuenciales.

        Args:
            quantum_state (List[complex]): Estado cuántico a procesar.

        Returns:
            Dict[str, Union[numpy.ndarray, float]]: Un diccionario que contiene las magnitudes y fases de la FFT, la entropía y la coherencia.

        Raises:
            ValueError: Si el estado cuántico está vacío.
            TypeError: Si el estado cuántico no es válido para conversión a numpy.array.
        """
        if not quantum_state:
            msg = "El estado cuántico no puede estar vacío."
            logger.error(msg)
            raise ValueError(msg)

        # Usa caché para evitar cálculos repetidos si el estado ya fue procesado.
        state_hash = hash(tuple(quantum_state))
        if state_hash in self.cache:
            return self.cache[state_hash]

        try:
            quantum_state_array = np.array(quantum_state, dtype=complex)  # Convierte a array de complejos.
        except Exception as e:
            logger.exception("Error al convertir el estado cuántico a np.array")
            raise TypeError("Estado cuántico inválido") from e

        fft_result = np.fft.fft(quantum_state_array)  # Aplica la FFT al estado cuántico.
        fft_magnitudes = np.abs(fft_result)  # Calcula las magnitudes de la FFT.
        fft_phases = np.angle(fft_result)  # Calcula las fases de la FFT.
        entropy = self.stat_analysis.shannon_entropy(fft_magnitudes.tolist())  # Calcula la entropía de Shannon.
        phase_variance = np.var(fft_phases)  # Calcula la varianza de las fases.
        coherence = np.exp(-phase_variance)  # Deriva una medida de coherencia a partir de la varianza.

        result = {
            'magnitudes': fft_magnitudes,
            'phases': fft_phases,
            'entropy': entropy,
            'coherence': coherence
        }
        self.cache[state_hash] = result  # Almacena el resultado en la caché.
        return result

    def fft_based_initializer(self, quantum_state: List[complex], out_dimension: int, scale: float = 0.01) -> torch.Tensor:
        """
        Inicializa una matriz de pesos basada en la FFT del estado cuántico.

        Args:
            quantum_state (List[complex]): Estado cuántico a procesar.
            out_dimension (int): Dimensión de salida de la matriz de pesos.
            scale (float): Factor de escala para los valores de los pesos. Por defecto 0.01.

        Returns:
            torch.Tensor: Matriz de pesos inicializada.
        """
        signal = np.array(quantum_state)  # Convierte el estado cuántico en un array.
        fft_result = np.fft.fft(signal)  # Aplica la FFT.
        magnitudes = np.abs(fft_result)  # Obtiene las magnitudes.
        norm_magnitudes = magnitudes / np.sum(magnitudes)  # Normaliza las magnitudes.
        weight_matrix = scale * np.tile(norm_magnitudes, (out_dimension, 1))  # Crea matriz replicando el vector.
        return torch.tensor(weight_matrix, dtype=torch.float32)  # Convierte la matriz a tensor de PyTorch.

    def advanced_fft_initializer(self, quantum_state: List[complex], out_dimension: int,
                                in_dimension: Optional[int] = None, scale: float = 0.01,
                                use_phases: bool = True) -> torch.Tensor:
        """
        Inicializador avanzado que crea una matriz rectangular utilizando magnitudes y fases de la FFT.

        Args:
            quantum_state (List[complex]): Estado cuántico a procesar.
            out_dimension (int): Dimensión de salida de la matriz de pesos.
            in_dimension (Optional[int]): Dimensión de entrada de la matriz de pesos. Si es None, se usa la longitud del estado cuántico.
            scale (float): Factor de escala para los valores de los pesos. Por defecto 0.01.
            use_phases (bool): Si se deben usar las fases en la inicialización. Por defecto True.

        Returns:
            torch.Tensor: Matriz de pesos inicializada.
        """
        signal = np.array(quantum_state)
        in_dimension = in_dimension or len(quantum_state)  # Define la dimensión de entrada si no se especifica.
        fft_result = np.fft.fft(signal)
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        norm_magnitudes = magnitudes / np.sum(magnitudes)  # Normaliza las magnitudes.

        # Construye el vector base para la matriz tomando la cantidad adecuada de características.
        if len(quantum_state) >= in_dimension:
            base_features = norm_magnitudes[:in_dimension]
        else:
            repeats = int(np.ceil(in_dimension / len(quantum_state)))
            base_features = np.tile(norm_magnitudes, repeats)[:in_dimension]

        if use_phases:
            # Incorpora la información de fase en las características.
            if len(quantum_state) >= in_dimension:
                phase_features = phases[:in_dimension]
            else:
                repeats = int(np.ceil(in_dimension / len(quantum_state)))
                phase_features = np.tile(phases, repeats)[:in_dimension]
            base_features = base_features * (1 + 0.1 * np.cos(phase_features))

        # Crea la matriz de pesos desplazando el vector base para cada fila.
        weight_matrix = np.empty((out_dimension, in_dimension))
        for i in range(out_dimension):
            shift = i % len(base_features)
            weight_matrix[i] = np.roll(base_features, shift)
        weight_matrix = scale * weight_matrix / np.max(np.abs(weight_matrix))  # Escala la matriz.
        return torch.tensor(weight_matrix, dtype=torch.float32)


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de prueba
    sample_data = [1, 2, 3, 4, 5, 5, 2]
    stat_analysis = StatisticalAnalysis()
    entropy = stat_analysis.shannon_entropy(sample_data)
    env_value = 0.8  # Ejemplo de valor de entorno

    cos_x, cos_y, cos_z = stat_analysis.calculate_cosines(entropy, env_value)

    print(f"Entropía: {entropy:.4f}")
    print(f"Cosenos direccionales: cos_x = {cos_x:.4f}, cos_y = {cos_y:.4f}, cos_z = {cos_z:.4f}")

    # Crear instancia de PRN
    prn = PRN(0.7, "frecuencial", alpha=0.5, beta=1.2)
    print(f"PRN creado: {prn}")

    # Crear PRN complejo
    complex_prn = ComplexPRN(0.6, 0.8, "wavelet", threshold=0.3)
    print(f"PRN complejo creado: {complex_prn}")
    print(f"Fase del PRN complejo: {complex_prn.get_phase():.4f} radianes")

    # Crear PRN mejorado
    enhanced_prn = EnhancedPRN(0.5, "quantum", decay=0.01)
    print(f"PRN mejorado creado: {enhanced_prn}")

    # Crear un circuito cuántico simulado
    circuit = ResilientQuantumCircuit(2)  # 2 qubits

    # Crear integrador FFT-Bayes
    integrator = FFTBayesIntegrator()

    # Procesar el circuito
    result = integrator.process_quantum_circuit(circuit)
    print(f"Entropía del estado del circuito: {result['entropy']:.4f}")
    print(f"Coherencia del estado del circuito: {result['coherence']:.4f}")

    # Inicializar pesos con FFT
    weights = integrator.advanced_fft_initializer(circuit.get_complex_amplitudes(), 4, 8)
    print(f"Forma de la matriz de pesos: {weights.shape}")
