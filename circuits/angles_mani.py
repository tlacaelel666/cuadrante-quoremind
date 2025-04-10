#!/usr/bin/env python3
"""
Módulo: IntegratedQuantumNetwork

Este módulo integra circuitos cuánticos resistentes con análisis FFT y bayesiano.
Combina la creación de estados cuánticos mediante Qiskit con procesamiento
avanzado basado en FFT para generar features e inicializar redes neuronales.

Autor: Jacobo Tlacaelel Mina Rodríguez. (optimizado y documentación por Gemini, Claude y ChatGPT).
Fecha creación/última actualización: 16/03/2025-08/04/2025.
Versión: cuadrante-coremind v1.2.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.covariance import EmpiricalCovariance
import logging
import os
import json

# Configuración del logger
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            logger.warning(f"Influencia ajustada a {max(0, min(1, new_influence))} para mantenerla en el rango [0,1]")
            new_influence = max(0, min(1, new_influence))

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

        # Considerar la combinación de parámetros de forma más inteligente
        combined_params = self.parameters.copy()
        for key, value in other_prn.parameters.items():
            if key in combined_params:
                if isinstance(combined_params[key], (int, float)) and isinstance(value, (int, float)):
                    combined_params[key] = combined_params[key] * weight + value * (1 - weight)
                elif isinstance(combined_params[key], list) and isinstance(value, list):
                    combined_params[key] = list(set(combined_params[key] + value)) # Unir listas sin duplicados
                elif isinstance(combined_params[key], dict) and isinstance(value, dict):
                    combined_params[key] = {**combined_params[key], **value} # Unir diccionarios
                else:
                    logger.warning(f"Conflicto de tipos al combinar parámetros '{key}'. Usando el valor del primer PRN.")
            else:
                combined_params[key] = value

        # Elegir el tipo de algoritmo según el peso (se podría refinar)
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

    def save_prn(self, filepath: str) -> None:
        """Guarda el objeto PRN en un archivo JSON."""
        data = {
            'influence': self.influence,
            'algorithm_type': self.algorithm_type,
            'parameters': self.parameters
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Objeto PRN guardado en: {filepath}")
        except Exception as e:
            logger.error(f"Error al guardar el objeto PRN: {e}")

    @classmethod
    def load_prn(cls, filepath: str) -> 'PRN':
        """Carga un objeto PRN desde un archivo JSON."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(influence=data['influence'], algorithm_type=data.get('algorithm_type'), **data.get('parameters', {}))
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error al cargar el objeto PRN: {e}")
            return None

    def __str__(self) -> str:
        """
        Representación en string del objeto PRN.

        Returns:
            str: Descripción del objeto PRN.
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm='{self.algorithm_type}'" if self.algorithm_type else ""
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
        algo_str = f", algorithm='{self.algorithm_type}'" if self.algorithm_type else ""
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

        if quantum_states.ndim != 2:
            logger.error(f"Se esperaba un array de estados cuánticos bidimensional, se recibió uno con {quantum_states.ndim} dimensiones.")
            return entropy, np.nan

        if quantum_states.shape[0] < 2:
            logger.warning("Se requieren al menos dos muestras para estimar la covarianza.")
            return entropy, np.nan

        try:
            # Estimación de la covarianza y cálculo de la distancia de Mahalanobis
            cov_estimator = EmpiricalCovariance().fit(quantum_states)
            mean_state = np.mean(quantum_states, axis=0)
            inv_cov = np.linalg.pinv(cov_estimator.covariance_)

            # Versión vectorizada del cálculo de la distancia usando np.einsum
            diff = quantum_states - mean_state
            aux = diff @ inv_cov
            sq_dists = np.einsum('ij,ij->i', aux, diff)
            distances = np.sqrt(sq_dists)
            mahal_mean = np.mean(distances)

            # Se registra la distancia promedio
            self.mahalanobis_records.append(mahal_mean)

            return entropy, mahal_mean
        except Exception as e:
            logger.error(f"Error al calcular la distancia de Mahalanobis: {e}")
            return entropy, np.nan


def mahalanobis_distance(data: np.ndarray, mean: np.ndarray, cov: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calcula la distancia de Mahalanobis para cada muestra en data.

    Args:
        data (np.ndarray): Array de datos con forma (n_muestras, n_features).
        mean (np.ndarray): Vector de media con forma (n_features,).
        cov (Optional[np.ndarray]): Matriz de covarianza. Si no se proporciona,
                                     se estima a partir de data.

    Returns:
        np.ndarray: Array unidimensional con la distancia de Mahalanobis para cada muestra.

    Raises:
        ValueError: Si las dimensiones de los datos y la media no coinciden.
        ValueError: Si la matriz de covarianza tiene una forma incorrecta.
        np.linalg.LinAlgError: Si la matriz de covarianza no es invertible.
    """
    n_features = mean.shape[0]
    if data.shape[1] != n_features:
        raise ValueError(f"El número de características en los datos ({data.shape[1]}) debe coincidir con la dimensión de la media ({n_features}).")

    if cov is None:
        cov_estimator = EmpiricalCovariance().fit(data)
        cov = cov_estimator.covariance_
    elif cov.shape != (n_features, n_features):
        raise ValueError(f"La matriz de covarianza debe tener forma ({n_features}, {n_features}), pero tiene forma {cov.shape}.")

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        logger.error("La matriz de covarianza es singular (no invertible). Usando pseudoinversa.")
        inv_cov = np.linalg.pinv(cov)

    diff = data - mean
    # Cálculo vectorizado de la distancia
    left_product = diff @ inv_cov
    squared_distances = np.einsum('ij,ij->i', left_product, diff)
    return np.sqrt(squared_distances)


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
        if not data:
            return 0.0  # La entropía de un conjunto vacío es cero

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
        # Magnitud del vector tridimensional:
        magnitude = np.sqrt(entropy ** 2 + env_value ** 2 + 1)

        # Evitar división por cero directamente en la magnitud
        if magnitude == 0:
            logger.warning("La magnitud del vector es cero, retornando cosenos con valor cero.")
            return 0.0, 0.0, 0.0

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
        if not evidence:
            logger.warning("Se proporcionó evidencia vacía. La distribución posterior no se actualizará.")
            return self.posterior if self.posterior else self.prior if self.prior else {}

        # Si no hay prior, usar la evidencia como prior (normalizada)
        if not self.prior:
            total_evidence = sum(evidence.values())
            if total_evidence > 0:
                self.prior = {k: v / total_evidence for k, v in evidence.items()}
            else:
                logger.warning("La evidencia inicial tiene una suma de cero. No se puede establecer un prior.")
                return {}

        # Calcular normalización
        total = sum(self.prior.get(k, 0) * evidence.get(k, 0) for k in set(self.prior) | set(evidence))

        if total == 0:
            logger.warning("La probabilidad total después de aplicar la evidencia es cero. Revisar prior y evidencia.")
            return {}

        # Actualizar posterior
        self.posterior = {k: (self.prior.get(k, 0) * evidence.get(k, 0)) / total for k in set(self.prior) | set(evidence)}

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
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("El número de qubits debe ser un entero positivo.")
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
            Para una implementación más completa, considera usar bibliotecas como Qiskit.
        """
        if not isinstance(gate, np.ndarray) or gate.ndim != 2:
            raise ValueError("La compuerta debe ser una matriz cuadrada.")
        if gate.shape[0] != gate.shape[1]:
            raise ValueError("La compuerta debe ser una matriz cuadrada.")
        if any(q < 0 or q >= self.n_qubits for q in target_qubits):
            raise ValueError("Los qubits objetivo están fuera de rango.")
        # Aquí iría la lógica para aplicar la compuerta usando productos tensoriales.
        # Esta es una versión simplificada para ilustrar la estructura.
        logger.warning("La aplicación de compuertas cuánticas es una implementación simplificada.")
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
        return {format(i, f"0{self.n_qubits}b"): prob.item() for i, prob in enumerate(probs)} # .item() para asegurar que sean floats estándar


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
            Dict[str, Union[np.ndarray, float]]: Un diccionario que contiene las magnitudes y fases de la FFT, la entropía y la coherencia.
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
            logger.debug(f"Se encontró el estado cuántico en la caché (hash: {state_hash}).")
            return self.cache[state_hash]

        try:
            quantum_state_array = np.array(quantum_state, dtype=complex)  # Convierte a array de complejos.
        except Exception as e:
            logger.exception("Error al convertir el estado cuántico a np.array")
            raise TypeError("Estado cuántico inválido") from e

        n_points = quantum_state_array.shape[0]
        if n_points > 0:
            fft_result = np.fft.fft(quantum_state_array)  # Aplica la FFT al estado cuántico.
            fft_magnitudes = np.abs(fft_result)  # Calcula las magnitudes de la FFT.
            fft_phases = np.angle(fft_result)  # Calcula las fases de la FFT.
            entropy = self.stat_analysis.shannon_entropy(fft_magnitudes.tolist())  # Calcula la entropía de Shannon.
            phase_variance = np.var(fft_phases) if len(fft_phases) > 1 else 0.0 # Evitar error si solo hay un punto.
            coherence = np.exp(-phase_variance)  # Deriva una medida de coherencia a partir de la varianza.

            result = {
                'magnitudes': fft_magnitudes,
                'phases': fft_phases,
                'entropy': entropy,
                'coherence': coherence
            }
            self.cache[state_hash] = result  # Almacena el resultado en la caché.
            logger.debug(f"Estado cuántico procesado y almacenado en caché (hash: {state_hash}).")
            return result
        else:
            logger.warning("El estado cuántico tiene cero puntos, retornando valores por defecto.")
            return {'magnitudes': np.array([]), 'phases': np.array([]), 'entropy': 0.0, 'coherence': 1.0}


    def fft_based_initializer(self, quantum_state: List[complex], out_dimension: int, scale: float = 0.01) -> torch.Tensor:
        """
        Inicializa una matriz de pesos basada en la FFT del estado cuántico.

        Args:
            quantum_state (List[complex]): Estado cuántico a procesar.
            out_dimension (int): Dimensión de salida de la matriz de pesos.
            scale (float): Factor de escala para los valores de los pesos.
        """
        fft_features = self.process_quantum_state(quantum_state)['magnitudes']
        n_features = len(fft_features)

        if n_features == 0:
            logger.warning("No se encontraron características de la FFT. Retornando tensor de ceros.")
            return torch.zeros(out_dimension, n_features)

        # Asegurar que las dimensiones sean compatibles
        if n_features < out_dimension:
            # Rellenar con ceros si es necesario
            padding = np.zeros(out_dimension - n_features)
            padded_features = np.concatenate((fft_features, padding))
            weight_matrix = torch.tensor(padded_features, dtype=torch.float).unsqueeze(0).T * scale # out_dimension x 1
        elif n_features > out_dimension:
            # Tomar las primeras 'out_dimension' características
            weight_matrix = torch.tensor(fft_features[:out_dimension], dtype=torch.float).unsqueeze(0).T * scale # out_dimension x 1
        else:
            weight_matrix = torch.tensor(fft_features, dtype=torch.float).unsqueeze(0).T * scale # out_dimension x 1

        return weight_matrix # Retorna una matriz de pesos de dimensión out_dimension x 1

# Ejemplo de uso (podrías mover esto a un bloque `if __name__ == "__main__":`)
if __name__ == "__main__":
    # Configurar nivel de logging más detallado para pruebas
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    # Ejemplo de PRN
    prn1 = PRN(influence=0.8, algorithm_type="genetic", mutation_rate=0.05)
    prn2 = PRN(influence=0.3, algorithm_type="annealing", temperature=100)
    print(prn1)
    print(prn2)
    prn_combined = prn1.combine_with(prn2, weight=0.7)
    print(f"Combinado con peso 0.7: {prn_combined}")
    prn_combined.adjust_influence(0.15)
    print(f"Ajustado: {prn_combined}")
    prn1.save_prn("prn1.json")
    loaded_prn = PRN.load_prn("prn1.json")
    print(f"Cargado desde archivo: {loaded_prn}")

    # Ejemplo de ComplexPRN
    complex_prn = ComplexPRN(real_component=1.0, imaginary_component=1.0)
    print(complex_prn)
    print(f"Fase de ComplexPRN: {complex_prn.get_phase()}")

    # Ejemplo de EnhancedPRN
    enhanced_prn = EnhancedPRN(influence=0.6)
    probabilities = {"00": 0.6, "01": 0.3, "10": 0.05, "11": 0.05}
    quantum_states = np.array([[1, 0], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)], [1, 1]])
    entropy, mahal_mean = enhanced_prn.record_quantum_noise(probabilities, quantum_states)
    print(f"Entropía: {entropy}, Distancia Mahalanobis promedio: {mahal_mean}")
    print(f"Registros Mahalanobis: {enhanced_prn.mahalanobis_records}")

    # Ejemplo de StatisticalAnalysis
    stats = StatisticalAnalysis()
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    entropy_data = stats.shannon_entropy(data)
    print(f"Entropía de los datos: {entropy_data}")
    cos_x, cos_y, cos_z = stats.calculate_cosines(entropy=entropy_data, env_value=0.5)
    print(f"Cosenos direccionales: ({cos_x:.4f}, {cos_y:.4f}, {cos_z:.4f})")

    # Ejemplo de BayesLogic
    bayes = BayesLogic(prior={"A": 0.6, "B": 0.4})
    evidence = {"A": 0.8, "B": 0.2}
    posterior = bayes.update(evidence)
    print(f"Posterior: {posterior}")
    max_posterior, prob = bayes.get_maximum_posterior()
    print(f"Máxima probabilidad posterior: {max_posterior} con probabilidad {prob:.4f}")

    bayes2 = BayesLogic()
    evidence2 = {"X": 0.7, "Y": 0.3}
    posterior2 = bayes2.update(evidence2)
    print(f"Posterior sin prior inicial: {posterior2}")

    # Ejemplo de ResilientQuantumCircuit
    circuit = ResilientQuantumCircuit(n_qubits=2)
    print(f"Estado inicial: {circuit.get_probabilities()}")
    # circuit.apply_gate(...) # Necesitarías implementar la lógica de las compuertas
    amplitudes = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)] # Ejemplo de estado después de operaciones
    circuit.state_vector = np.array(amplitudes)
    print(f"Estado después de operaciones: {circuit.get_probabilities()}")
    print(f"Amplitudes complejas: {circuit.get_complex_amplitudes()}")

    # Ejemplo de FFTBayesIntegrator
    integrator = FFTBayesIntegrator()
    processed_state = integrator.process_quantum_state(circuit.get_complex_amplitudes())
    print(f"Resultado del procesamiento FFT: {processed_state.keys()}")
    print(f"Magnitudes FFT: {processed_state['magnitudes']}")
    print(f"Entropía FFT: {processed_state['entropy']}")
    print(f"Coherencia FFT: {processed_state['coherence']}")

    initial_weights = integrator.fft_based_initializer(circuit.get_complex_amplitudes(), out_dimension=4)
    print(f"Pesos iniciales basados en FFT (shape: {initial_weights.shape}):\n{initial_weights}")
