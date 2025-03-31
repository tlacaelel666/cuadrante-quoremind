#!/usr/bin/env python3
"""
Módulo Híbrido de Integración Neuronal Cuántico-Clásico

# quantum_hybrid_system.py

Este módulo define un sistema completo que integra:
  • Una red neural clásica (NeuralNetwork) con múltiples funciones de activación y
    soporte para optimizadores (SGD y Adam).
  • Un activador cuántico-clásico (QuantumClassicalActivator) que crea circuitos cuánticos
    relacionados con funciones de activación y permite realizar un paso forward híbrido.
  • Un sistema de estado cuántico (QuantumState) que utiliza métodos bayesianos y distancias
    de Mahalanobis para gestionar y predecir estados cuánticos simulados.
  • (Nota: La clase QuantumNeuralHybridSystem no está presente en este fragmento,
    pero se infiere como un posible componente integrador)

Autor: Jacobo Tlacaelel Mina Rodríguez
Fecha: 13/03/2025 (Ajustada a la fecha actual de interacción)
Versión: cuadrante-coremind v1.0
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import joblib
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import logging
import sys
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("init_activation.log"), # Descomentar para guardar en archivo
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Importaciones de Módulos Personalizados (Asumiendo estructura de archivos) ---
# Intentar importar BayesLogic y funciones estadísticas
try:
    # Se asume que bayes_logic.py está en el mismo directorio o en el PYTHONPATH
    from bayes_logic import BayesLogic, shannon_entropy, calculate_cosines, compute_mahalanobis_distance
    logger.info("Módulo bayes_logic importado correctamente.")
except ImportError as e:
    logger.error(f"Error al importar desde bayes_logic.py: {e}. Usando implementaciones de respaldo.")

    # --- Implementaciones de Respaldo (si bayes_logic.py no está) ---
    class BayesLogic:
        """Clase de respaldo para BayesLogic."""
        def __init__(self) -> None:
            self.EPSILON = 1e-6
            self.HIGH_ENTROPY_THRESHOLD = 0.8
            self.HIGH_COHERENCE_THRESHOLD = 0.6
            self.ACTION_THRESHOLD = 0.5

        def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
            prior_b = prior_b if prior_b != 0 else self.EPSILON
            return (conditional_b_given_a * prior_a) / prior_b

        def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
            prior = prior if prior != 0 else self.EPSILON
            return joint_probability / prior

        def calculate_high_entropy_prior(self, entropy: float) -> float:
            return 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.1

        def calculate_high_coherence_prior(self, coherence: float) -> float:
            return 0.6 if coherence > self.HIGH_COHERENCE_THRESHOLD else 0.2

        def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
            if coherence > self.HIGH_COHERENCE_THRESHOLD:
                if action == 1:
                    return prn_influence * 0.8 + (1 - prn_influence) * 0.2
                else:
                    return prn_influence * 0.1 + (1 - prn_influence) * 0.7
            return 0.3

        def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, prn_influence: float,
                                                      action: int) -> Dict[str, float]:
            high_entropy_prior = self.calculate_high_entropy_prior(entropy)
            high_coherence_prior = self.calculate_high_coherence_prior(coherence)
            conditional_b_given_a = (prn_influence * 0.7 + (1 - prn_influence) * 0.3
                                     if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.2)
            posterior_a_given_b = self.calculate_posterior_probability(high_entropy_prior, high_coherence_prior, conditional_b_given_a)
            joint_probability_ab = self.calculate_joint_probability(coherence, action, prn_influence)
            conditional_action_given_b = self.calculate_conditional_probability(joint_probability_ab, high_coherence_prior)
            action_to_take = 1 if conditional_action_given_b > self.ACTION_THRESHOLD else 0

            return {
                "action_to_take": action_to_take,
                "high_entropy_prior": high_entropy_prior,
                "high_coherence_prior": high_coherence_prior,
                "posterior_a_given_b": posterior_a_given_b,
                "conditional_action_given_b": conditional_action_given_b
            }

    def shannon_entropy(data: List[float]) -> float:
        """Respaldo: Calcula la entropía de Shannon."""
        values, counts = np.unique(np.round(data, decimals=6), return_counts=True)
        if len(data) == 0: return 0.0
        probabilities = counts / len(data)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_cosines(entropy: float, prn_object: float) -> Tuple[float, float, float]:
        """Respaldo: Calcula cosenos directores."""
        entropy = max(entropy, 1e-6)
        prn_object = max(prn_object, 1e-6)
        magnitude = np.sqrt(entropy ** 2 + prn_object ** 2 + 1)
        cos_x = entropy / magnitude
        cos_y = prn_object / magnitude
        cos_z = 1 / magnitude
        return cos_x, cos_y, cos_z

    def compute_mahalanobis_distance(data: List[List[float]], point: List[float]) -> float:
        """Respaldo: Calcula la distancia de Mahalanobis."""
        data_array = np.array(data)
        point_array = np.array(point)
        if data_array.shape[0] < data_array.shape[1]: return np.inf # No se puede calcular covarianza
        try:
            covariance_estimator = EmpiricalCovariance().fit(data_array)
            cov_matrix = covariance_estimator.covariance_
            inv_cov_matrix = np.linalg.pinv(cov_matrix) # Usar pinv para robustez
            mean_vector = np.mean(data_array, axis=0)
            distance = mahalanobis(point_array, mean_vector, inv_cov_matrix)
            return distance
        except Exception:
            return np.inf # Fallback

# Re-importar PRN de la definición anterior si es necesario
try:
    # Asumiendo que PRN está definido en bayes_logic o en el script anterior
    # Si no, deberías copiar/pegar la definición de la clase PRN aquí también.
    # from bayes_logic import PRN # O donde esté definido
    # Placeholder si PRN no se importa:
    class PRN:
        def __init__(self, influence: float = 0.5, **kwargs):
             self.influence = max(0, min(1, influence))
        def record_noise(self, probs): # Método dummy si no existe
            return shannon_entropy(list(probs.values()))
    logger.info("Clase PRN (o placeholder) disponible.")
except ImportError:
     logger.warning("Clase PRN no encontrada, usando placeholder.")
     class PRN:
        def __init__(self, influence: float = 0.5, **kwargs):
             self.influence = max(0, min(1, influence))
        def record_noise(self, probs): # Método dummy si no existe
            return shannon_entropy(list(probs.values()))


# --- Clases del Módulo `quantum_bayes_mahalanobis.py` (Reutilizadas aquí) ---

class QuantumBayesMahalanobis(BayesLogic):
    """
    Clase que combina la lógica de Bayes con el cálculo de la distancia de Mahalanobis
    aplicada a estados cuánticos (simulados), permitiendo proyecciones vectorizadas
    e inferencias de coherencia/entropía.
    """
    def __init__(self):
        super().__init__()
        self.covariance_estimator = EmpiricalCovariance()

    def _get_inverse_covariance(self, data: np.ndarray) -> np.ndarray:
        """Calcula la (pseudo)inversa de la matriz de covarianza."""
        if data.ndim != 2 or data.shape[0] <= data.shape[1]:
             logger.warning("Datos insuficientes o inválidos para calcular covarianza robusta. Usando identidad.")
             return np.identity(data.shape[1]) # Fallback a matriz identidad
        try:
            self.covariance_estimator.fit(data)
            cov_matrix = self.covariance_estimator.covariance_
            inv_cov_matrix = np.linalg.pinv(cov_matrix) # Usar pinv para robustez
        except Exception as e:
            logger.error(f"Error calculando inversa de covarianza: {e}. Usando identidad.")
            inv_cov_matrix = np.identity(data.shape[1])
        return inv_cov_matrix

    def compute_quantum_mahalanobis(self, quantum_states_A: np.ndarray, quantum_states_B: np.ndarray) -> np.ndarray:
        """Calcula la distancia de Mahalanobis vectorizada."""
        if quantum_states_A.ndim != 2 or quantum_states_B.ndim != 2:
            raise ValueError("Los estados cuánticos deben ser matrices bidimensionales.")
        if quantum_states_A.shape[1] != quantum_states_B.shape[1]:
            raise ValueError("La dimensión (n_dimensiones) de A y B debe coincidir.")
        if quantum_states_A.shape[0] < 2: # Necesitamos al menos 2 puntos para covarianza
             logger.warning("Pocos datos en quantum_states_A para Mahalanobis. Devolviendo distancia Euclidiana.")
             mean_A = np.mean(quantum_states_A, axis=0) if quantum_states_A.shape[0] > 0 else np.zeros(quantum_states_B.shape[1])
             diff_B = quantum_states_B - mean_A
             distances = np.sqrt(np.sum(diff_B**2, axis=1))
             return distances

        inv_cov_matrix = self._get_inverse_covariance(quantum_states_A)
        mean_A = np.mean(quantum_states_A, axis=0)

        diff_B = quantum_states_B - mean_A
        try:
            aux = diff_B @ inv_cov_matrix
            dist_sqr = np.einsum('ij,ij->i', aux, diff_B)
            # Evitar raíces negativas por errores numéricos
            dist_sqr = np.maximum(dist_sqr, 0)
            distances = np.sqrt(dist_sqr)
        except Exception as e:
             logger.error(f"Error en cálculo de distancia Mahalanobis: {e}. Devolviendo distancia Euclidiana.")
             distances = np.sqrt(np.sum(diff_B**2, axis=1))

        return distances

    def quantum_cosine_projection(self, quantum_states: np.ndarray, entropy: float, coherence: float) -> tf.Tensor:
        """Proyecta estados y calcula distancias Mahalanobis normalizadas."""
        if quantum_states.ndim == 1: # Si es un solo estado, convertir a matriz
             quantum_states = quantum_states.reshape(1, -1)
        if quantum_states.shape[1] != 2:
            # Si no tiene 2 columnas, usar las primeras 2 o duplicar si solo hay 1
            if quantum_states.shape[1] == 1:
                 quantum_states = np.hstack([quantum_states, quantum_states])
                 logger.warning("Se duplicó la única columna de quantum_states para proyección.")
            elif quantum_states.shape[1] > 2:
                 quantum_states = quantum_states[:, :2]
                 logger.warning("Se usaron solo las primeras 2 columnas de quantum_states para proyección.")
            else: # shape[1] == 0
                 raise ValueError("quantum_states no tiene columnas para proyección.")


        cos_x, cos_y, cos_z = calculate_cosines(entropy, coherence)

        projected_states_A = quantum_states * np.array([cos_x, cos_y])
        projected_states_B = quantum_states * np.array([cos_x * cos_z, cos_y * cos_z])

        # Necesitamos al menos 2 puntos para la referencia A
        if projected_states_A.shape[0] < 2:
             ref_A = np.vstack([projected_states_A, projected_states_A + 1e-6]) # Añadir punto artificial
        else:
             ref_A = projected_states_A

        mahalanobis_distances = self.compute_quantum_mahalanobis(ref_A, projected_states_B)

        mahalanobis_distances_tf = tf.convert_to_tensor(mahalanobis_distances, dtype=tf.float32)
        # Aplicar Softmax solo si hay más de un valor, sino devolver 1.0
        if tf.size(mahalanobis_distances_tf) > 1:
            normalized_distances = tf.nn.softmax(mahalanobis_distances_tf)
        elif tf.size(mahalanobis_distances_tf) == 1:
             normalized_distances = tf.constant([1.0], dtype=tf.float32)
        else: # Vacío
             normalized_distances = tf.constant([], dtype=tf.float32)

        return normalized_distances

    def calculate_quantum_posterior_with_mahalanobis(self, quantum_states: np.ndarray, entropy: float, coherence: float) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calcula posterior y proyecciones."""
        quantum_projections = self.quantum_cosine_projection(quantum_states, entropy, coherence)

        if tf.size(quantum_projections) == 0: # Si no hay proyecciones, devolver valores neutros
             return tf.constant(0.5, dtype=tf.float32), quantum_projections

        # Calcular covarianza si hay más de una proyección
        if tf.size(quantum_projections) > 1:
            # Necesita ser 2D para tfp.stats.covariance
            tensor_projections_2d = tf.expand_dims(quantum_projections, axis=-1)
            quantum_covariance = tfp.stats.covariance(tensor_projections_2d, sample_axis=0)
            quantum_prior = tf.linalg.trace(quantum_covariance) # Traza de matriz 1x1 es el elemento
        else: # Si solo hay una proyección, la covarianza es 0
             quantum_prior = tf.constant(0.0, dtype=tf.float32)


        prior_coherence = self.calculate_high_coherence_prior(coherence)
        joint_prob = self.calculate_joint_probability(coherence, 1, tf.reduce_mean(quantum_projections).numpy())
        # Asegurar que quantum_prior no sea 0 para calcular cond_prob
        quantum_prior_safe = tf.maximum(quantum_prior, self.EPSILON)
        cond_prob = self.calculate_conditional_probability(joint_prob, quantum_prior_safe.numpy())

        # Asegurar que prior_coherence no sea 0 para calcular posterior
        prior_coherence_safe = max(prior_coherence, self.EPSILON)
        posterior = self.calculate_posterior_probability(quantum_prior_safe.numpy(), prior_coherence_safe, cond_prob)

        # Asegurarse que posterior sea un tensor float32
        posterior_tf = tf.convert_to_tensor(posterior, dtype=tf.float32)

        return posterior_tf, quantum_projections

    def predict_quantum_state(self, quantum_states: np.ndarray, entropy: float, coherence: float) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predice el siguiente estado cuántico (simulado)."""
        posterior, projections = self.calculate_quantum_posterior_with_mahalanobis(quantum_states, entropy, coherence)

        if tf.size(projections) == 0: # Manejar caso vacío
            return tf.constant(0.0, dtype=tf.float32), posterior

        # Ponderar las proyecciones por la posterior escalar
        # tf.expand_dims(posterior, -1) añade una dimensión para broadcasting si es necesario, aunque aquí posterior es escalar
        # Asegurarnos que projections sea float32
        projections_float32 = tf.cast(projections, tf.float32)
        # La predicción es una suma ponderada (escalar * vector -> vector, luego suma -> escalar)
        next_state_prediction = tf.reduce_sum(projections_float32 * posterior)

        return next_state_prediction, posterior

class EnhancedPRN(PRN):
    """Extiende PRN para registrar distancias de Mahalanobis."""
    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        # Asumiendo que PRN base tiene __init__(influence, ...)
        # Si no, adaptar la llamada a super() o inicializar aquí
        # super().__init__(influence, algorithm_type, **parameters) # Si PRN base lo soporta
        self.influence = max(0, min(1, influence)) # Inicializar aquí si PRN base no lo hace
        self.algorithm_type = algorithm_type
        self.parameters = parameters
        self.mahalanobis_records = []

    def record_noise(self, probabilities: dict): # Método dummy necesario si no viene de PRN base
        """Placeholder para record_noise si no se hereda."""
        return shannon_entropy(list(probabilities.values()))

    def record_quantum_noise(self, probabilities: dict, quantum_states: np.ndarray) -> Tuple[float, float]:
        """Registra ruido cuántico (simulado) basado en Mahalanobis."""
        entropy = self.record_noise(probabilities)

        if quantum_states.ndim != 2 or quantum_states.shape[0] < 2:
            logger.warning("Datos insuficientes para Mahalanobis en record_quantum_noise. Devolviendo 0.")
            mahal_mean = 0.0
        else:
            # Usar compute_mahalanobis_distance local o importado
            # Necesita un punto de referencia, usamos la media como punto para calcular distancia promedio
            mean_state = np.mean(quantum_states, axis=0)
            # Calcular distancia de cada punto a la media, usando la covarianza del dataset
            distances = compute_mahalanobis_distance(quantum_states, mean_state) # Asumiendo que compute_mahalanobis_distance está disponible
            mahal_mean = np.mean(distances) if distances is not np.inf else 0.0 # Tomar la media

        self.mahalanobis_records.append(mahal_mean)
        return entropy, mahal_mean


class QuantumNoiseCollapse(QuantumBayesMahalanobis):
    """Simula colapso de onda usando Mahalanobis como ruido."""
    def __init__(self, prn_influence: float = 0.5):
        super().__init__()
        self.prn = EnhancedPRN(influence=prn_influence)

    def simulate_wave_collapse(self, quantum_states: np.ndarray, prn_influence: float, previous_action: int) -> Dict:
        """Simula colapso de onda."""
        if quantum_states.ndim == 1: quantum_states = quantum_states.reshape(1, -1) # Asegurar 2D
        if quantum_states.shape[0] == 0: # Manejar entrada vacía
             logger.warning("quantum_states vacío en simulate_wave_collapse.")
             return { "collapsed_state": tf.constant(0.0), "action": previous_action, "entropy": 0.0,
                      "coherence": 0.0, "mahalanobis_distance": 0.0, "cosines": (0,0,1)}

        # Usar la media de cada estado como "probabilidad" para simplificar
        probabilities = {str(i): np.mean(np.abs(state)) for i, state in enumerate(quantum_states)}

        entropy, mahalanobis_mean = self.prn.record_quantum_noise(probabilities, quantum_states)
        cos_x, cos_y, cos_z = calculate_cosines(entropy, mahalanobis_mean)
        # Calcular coherencia heurísticamente
        coherence = max(0, min(1, np.exp(-mahalanobis_mean) * (cos_x + cos_y + cos_z) / 3.0))

        bayes_probs = self.calculate_probabilities_and_select_action(
            entropy=entropy, coherence=coherence, prn_influence=prn_influence, action=previous_action
        )

        projected_states_tf = self.quantum_cosine_projection(quantum_states, entropy, coherence)

        # Colapso: Ponderar proyección por acción (simplificado)
        collapsed_state_scalar = tf.reduce_sum(projected_states_tf * tf.cast(bayes_probs["action_to_take"], tf.float32))

        return {
            "collapsed_state": collapsed_state_scalar, # Ahora es un escalar
            "action": bayes_probs["action_to_take"],
            "entropy": entropy,
            "coherence": coherence,
            "mahalanobis_distance": mahalanobis_mean,
            "cosines": (cos_x, cos_y, cos_z)
        }

    def objective_function_with_noise(self, quantum_states: np.ndarray, target_state: np.ndarray, entropy_weight: float = 1.0) -> tf.Tensor:
        """Función objetivo combinando fidelidad y ruido simulado."""
        if quantum_states.ndim == 1: quantum_states = quantum_states.reshape(1,-1)
        if target_state.ndim == 1: target_state = target_state.reshape(1,-1)
        if quantum_states.shape[1] != target_state.shape[1]:
             raise ValueError("Dimensiones incompatibles entre quantum_states y target_state")
        if quantum_states.shape[0] == 0: return tf.constant(np.inf, dtype=tf.float32)

        # Fidelidad simple (solapamiento promedio)
        # Normalizar estados para que sean vectores unitarios (si no lo son ya)
        norm_qs = tf.linalg.normalize(tf.cast(quantum_states, tf.float32), axis=1)[0]
        norm_ts = tf.linalg.normalize(tf.cast(target_state, tf.float32), axis=1)[0]
        overlap = tf.abs(tf.tensordot(tf.math.conj(norm_qs), norm_ts, axes=1))
        fidelity = overlap**2

        probabilities = {str(i): np.mean(np.abs(st)) for i, st in enumerate(quantum_states)}
        entropy, mahalanobis_dist = self.prn.record_quantum_noise(probabilities, quantum_states)

        objective_value = ((1.0 - fidelity)
                           + entropy_weight * tf.cast(entropy, tf.float32)
                           + tf.cast((1.0 - np.exp(-mahalanobis_dist)), tf.float32))

        return objective_value

    def optimize_quantum_state(self, initial_states: np.ndarray, target_state: np.ndarray, max_iterations: int = 100, learning_rate: float = 0.01) -> Tuple[np.ndarray, float]:
        """Optimiza estados cuánticos (simulados) usando descenso de gradiente."""
        if initial_states.ndim == 1: initial_states = initial_states.reshape(1,-1)
        if target_state.ndim == 1: target_state = target_state.reshape(1,-1)

        current_states_var = tf.Variable(initial_states, dtype=tf.float32)
        target_state_const = tf.constant(target_state, dtype=tf.float32) # Target es constante

        best_objective_val = tf.constant(np.inf, dtype=tf.float32)
        best_states_val = current_states_var.numpy().copy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for iteration in range(max_iterations):
            with tf.GradientTape() as tape:
                tape.watch(current_states_var)
                # La función objetivo ahora recibe tensores
                objective = self.objective_function_with_noise(current_states_var.numpy(), target_state_const.numpy())

            grads = tape.gradient(objective, [current_states_var])

            if grads is None or grads[0] is None:
                logger.warning(f"No se pudo calcular el gradiente en la iteración {iteration}. Deteniendo optimización.")
                break

            optimizer.apply_gradients(zip(grads, [current_states_var]))

            # Normalizar estados después de aplicar gradientes (opcional pero puede ser útil)
            # current_states_var.assign(tf.linalg.normalize(current_states_var, axis=1)[0])

            # Evaluar el nuevo objetivo
            new_objective = self.objective_function_with_noise(current_states_var.numpy(), target_state_const.numpy())

            if new_objective < best_objective_val:
                best_objective_val = new_objective
                best_states_val = current_states_var.numpy().copy()
            # Simple criterio de parada si el objetivo no mejora mucho (opcional)
            # if iteration > 10 and abs(new_objective - prev_objective) < 1e-5:
            #    logger.info("Convergencia temprana detectada.")
            #    break
            # prev_objective = new_objective

        return best_states_val, best_objective_val.numpy()


# -------------------------
# Funciones de activación clásicas
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip para evitar overflow

def sigmoid_derivative(s: np.ndarray) -> np.ndarray: # Derivada en términos de la salida sigmoide s
    return s * (1 - s)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(t: np.ndarray) -> np.ndarray: # Derivada en términos de la salida tanh t
    return 1 - t**2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray: # Derivada en términos de la entrada x
    return np.where(x > 0, 1.0, 0.0)

# -------------------------
# Red neuronal clásica
# -------------------------
class NeuralNetwork:
    """Red Neuronal Clásica con soporte para SGD y Adam."""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, activation: str = "sigmoid"):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.num_layers = len(hidden_sizes) + 1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Inicialización de pesos y sesgos (Xavier/Glorot)
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(self.num_layers):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

        # Seleccionar funciones de activación
        self._select_activation_function()

        # Inicializar estado para Adam si se usa
        self.m_w: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        self.v_w: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        self.m_b: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        self.v_b: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        self.t_adam = 0 # Contador de pasos para Adam

    def _select_activation_function(self):
        """Selecciona las funciones de activación y sus derivadas."""
        if self.activation_name == "sigmoid":
            self.activate = sigmoid
            self.activate_derivative_from_output = sigmoid_derivative # Derivada respecto a la salida
        elif self.activation_name == "tanh":
            self.activate = tanh
            self.activate_derivative_from_output = tanh_derivative
        elif self.activation_name == "relu":
            self.activate = relu
            # La derivada de ReLU se calcula más fácilmente desde la entrada (o pre-activación Z)
            # Guardaremos Z en forward pass o recalcularemos si es necesario.
            # Por simplicidad aquí, definiremos una que necesita Z (pre-activación)
            self.activate_derivative_from_input = relu_derivative # Derivada respecto a la entrada (o Z)
        else:
            raise ValueError(f"Función de activación no reconocida: {self.activation_name}")

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Realiza el forward pass, guardando pre-activaciones (Z) y activaciones (A)."""
        activations_A = [X] # Lista de activaciones (A), A[0] = X
        pre_activations_Z = [] # Lista de pre-activaciones (Z)

        A = X
        for i in range(self.num_layers):
            W = self.weights[i]
            b = self.biases[i]
            Z = np.dot(A, W) + b
            pre_activations_Z.append(Z)
            A = self.activate(Z)
            activations_A.append(A)

        return pre_activations_Z, activations_A

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, optimizer: str = "adam"):
        """Realiza el backward pass y actualiza pesos."""
        m = X.shape[0] # Número de ejemplos
        pre_activations_Z, activations_A = self.forward(X)
        output_A = activations_A[-1]

        # Calcular delta para la capa de salida
        # Asumiendo error cuadrático medio: dError/dOutput = Output - y
        # dZ_output = dError/dOutput * dOutput/dZ_output
        delta_output = output_A - y
        if self.activation_name == "relu":
             # Necesitamos Z de la última capa
             dZ = delta_output * self.activate_derivative_from_input(pre_activations_Z[-1])
        else:
             # Derivada en términos de la salida A
             dZ = delta_output * self.activate_derivative_from_output(output_A)

        deltas = [dZ] # Lista de dZ para cada capa (de salida hacia entrada)

        # Propagar deltas hacia atrás
        for i in range(self.num_layers - 1, 0, -1):
             # delta[l] = delta[l+1] . W[l+1]^T * g'(Z[l])
             W_next = self.weights[i] # Pesos que conectan capa i con i+1
             dZ_prev_layer = deltas[-1] # dZ de la capa siguiente (l+1)

             if self.activation_name == "relu":
                  # Necesitamos Z de la capa actual i (guardado en pre_activations_Z[i-1])
                  dZ_curr_layer = np.dot(dZ_prev_layer, W_next.T) * self.activate_derivative_from_input(pre_activations_Z[i-1])
             else:
                  # Derivada en términos de la salida A de la capa actual i (guardado en activations_A[i])
                  dZ_curr_layer = np.dot(dZ_prev_layer, W_next.T) * self.activate_derivative_from_output(activations_A[i])

             deltas.append(dZ_curr_layer)

        deltas.reverse() # Ordenar deltas desde la primera capa oculta hasta la salida

        # Actualizar pesos y sesgos
        self.t_adam += 1 # Incrementar contador de pasos para Adam
        for i in range(self.num_layers):
            A_prev_layer = activations_A[i] # Activación de la capa anterior (o X para la primera capa)
            dZ_curr_layer = deltas[i] # dZ de la capa actual

            # Calcular gradientes dW y db
            dW = (1/m) * np.dot(A_prev_layer.T, dZ_curr_layer)
            db = (1/m) * np.sum(dZ_curr_layer, axis=0, keepdims=True)

            # Aplicar optimizador
            if optimizer == "sgd":
                self.weights[i] -= learning_rate * dW
                self.biases[i] -= learning_rate * db
            elif optimizer == "adam":
                self._adam_update(i, dW, db, learning_rate)
            else:
                raise ValueError(f"Optimizador no reconocido: {optimizer}")

    def _adam_update(self, layer_idx: int, dW: np.ndarray, db: np.ndarray, learning_rate: float,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Actualización de pesos y sesgos usando Adam."""
        t = self.t_adam

        # Actualizar momentos para pesos
        self.m_w[layer_idx] = beta1 * self.m_w[layer_idx] + (1 - beta1) * dW
        self.v_w[layer_idx] = beta2 * self.v_w[layer_idx] + (1 - beta2) * np.square(dW)
        # Corregir sesgo de momentos
        m_hat_w = self.m_w[layer_idx] / (1 - beta1**t)
        v_hat_w = self.v_w[layer_idx] / (1 - beta2**t)
        # Actualizar pesos
        self.weights[layer_idx] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)

        # Actualizar momentos para sesgos
        self.m_b[layer_idx] = beta1 * self.m_b[layer_idx] + (1 - beta1) * db
        self.v_b[layer_idx] = beta2 * self.v_b[layer_idx] + (1 - beta2) * np.square(db)
        # Corregir sesgo de momentos
        m_hat_b = self.m_b[layer_idx] / (1 - beta1**t)
        v_hat_b = self.v_b[layer_idx] / (1 - beta2**t)
        # Actualizar sesgos
        self.biases[layer_idx] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    def save_model(self, filepath: str):
        """Guarda los pesos y sesgos del modelo."""
        model_data = {'weights': self.weights, 'biases': self.biases}
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath: str):
        """Carga los pesos y sesgos del modelo."""
        try:
            model_data = joblib.load(filepath)
            if 'weights' in model_data and 'biases' in model_data:
                self.weights = model_data['weights']
                self.biases = model_data['biases']
                 # Reiniciar estado de Adam al cargar
                self.m_w = [np.zeros_like(w) for w in self.weights]
                self.v_w = [np.zeros_like(w) for w in self.weights]
                self.m_b = [np.zeros_like(b) for b in self.biases]
                self.v_b = [np.zeros_like(b) for b in self.biases]
                self.t_adam = 0
                logger.info(f"Modelo cargado desde: {filepath}")
            else:
                logger.error("Archivo de modelo inválido: faltan 'weights' o 'biases'.")
        except FileNotFoundError:
            logger.error(f"Archivo de modelo no encontrado: {filepath}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")


# -------------------------
# Tipos de activación para el componente cuántico-clásico
# -------------------------
class ActivationType(Enum):
    """Define los tipos de funciones de activación soportadas."""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"

# -------------------------
# Componente Híbrido: Activador Cuántico-Clásico
# -------------------------
class QuantumClassicalActivator:
    """
    Simula una activación híbrida que combina una función clásica
    con un circuito cuántico parametrizado asociado.
    """
    def __init__(self, n_qubits: int = 1): # Reducido a 1 qubit por simplicidad en circuitos ejemplo
        if n_qubits < 1:
            raise ValueError("Se necesita al menos 1 qubit.")
        self.n_qubits = n_qubits
        self._setup_activation_functions() # Configura las funciones clásicas

    def _setup_activation_function(self):
        """Configura funciones de activación y sus derivadas."""
        self.activation_functions = {
            ActivationType.SIGMOID: sigmoid,
            ActivationType.TANH: tanh,
            ActivationType.RELU: relu
        }
        # Nota: Las derivadas clásicas no se usan directamente en el forward híbrido,
        # pero podrían ser útiles para una retropropagación híbrida más compleja.
        self.activation_derivatives = {
            ActivationType.SIGMOID: sigmoid_derivative,
            ActivationType.TANH: tanh_derivative,
            ActivationType.RELU: relu_derivative # Usando la definición que depende de la entrada
        }
        logger.debug("Funciones de activación clásicas configuradas.")


    def create_quantum_activation_circuit(self, activation_type: ActivationType) -> Tuple[QuantumCircuit, Parameter]:
        """
        Crea un circuito cuántico parametrizado asociado a la función de activación.
        Estos son circuitos *ejemplo* y no necesariamente implementan la función clásica.
        Representan una componente cuántica que *podría* estar influenciada por la activación.

        Args:
            activation_type (ActivationType): Tipo de función de activación.

        Returns:
            Tuple[QuantumCircuit, Parameter]: Circuito cuántico generado y su parámetro.
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        # No necesitamos registro clásico si no medimos aquí
        qc = QuantumCircuit(qr, name=f"{activation_type.value}_qc")
        theta = Parameter('θ') # Parámetro para controlar la rotación

        if activation_type == ActivationType.SIGMOID:
            # Circuito simple: Rotación Y controlada por theta
            qc.ry(theta, qr[0]) # Rotación proporcional al parámetro theta
        elif activation_type == ActivationType.TANH:
            # Circuito simple: Rotación Z controlada por theta
            qc.rz(theta, qr[0])
        elif activation_type == ActivationType.RELU:
            # Circuito simple: Rotación X controlada por theta
            # Podría interpretarse como: si theta > 0 (análogo a x>0), aplicar rotación
            qc.rx(theta, qr[0])
        else:
             raise ValueError(f"Tipo de activación no soportado para circuito cuántico: {activation_type}")

        logger.debug(f"Circuito cuántico creado para activación: {activation_type.value}")
        return qc, theta

    def quantum_influenced_activation(self,
                                      x: float, # Asumimos entrada escalar para el parámetro cuántico
                                      activation_type: ActivationType,
                                      quantum_influence: float = 0.1
                                      ) -> float:
        """
        Calcula una activación influenciada por un factor cuántico simulado.
        Esta es una simulación conceptual, no ejecuta un circuito real.

        Args:
            x (float): Valor de pre-activación (usado para obtener activación clásica).
            activation_type (ActivationType): Tipo de activación.
            quantum_influence (float): Factor que determina cuánto afecta el "efecto cuántico".

        Returns:
            float: Valor de activación modificado.
        """
        # 1. Calcular activación clásica
        classic_activation_func = self.activation_functions[activation_type]
        classic_output = classic_activation_func(x)

        # 2. Simular un factor cuántico simple (ej. basado en seno)
        # El parámetro 'theta' del circuito podría estar relacionado con 'x'
        # Aquí simulamos un efecto oscilatorio simple como factor cuántico.
        theta_simulated = np.pi * classic_output # Relacionar theta con la salida clásica
        quantum_factor = np.sin(theta_simulated)**2 # Factor entre 0 y 1

        # 3. Combinar salida clásica con factor cuántico
        # El factor cuántico modula la salida clásica
        hybrid_output = classic_output * (1 - quantum_influence * quantum_factor)

        logger.debug(f"Activación híbrida: Clásica={classic_output:.4f}, FactorQ={quantum_factor:.4f}, Híbrida={hybrid_output:.4f}")
        return hybrid_output


# -------------------------
# Clase QuantumState (Simplificada para este contexto)
# -------------------------
class QuantumState(QuantumBayesMahalanobis):
    """
    Gestiona un estado cuántico simulado (vector de probabilidades clásico)
    utilizando métodos bayesianos y de Mahalanobis.
    (Reutiliza la implementación anterior con métodos adicionales)
    """
    def __init__(self, num_positions: int, learning_rate: float = 0.1):
        super().__init__()
        if num_positions <= 0:
            raise ValueError("num_positions debe ser positivo.")
        self.num_positions = num_positions
        self.learning_rate = learning_rate
        # Inicializar estado con distribución más uniforme
        self.state_vector = np.ones(num_positions) / np.sqrt(num_positions)
        # self.state_vector = self.normalize_state(np.random.rand(num_positions)) # Alternativa aleatoria
        self.probabilities = np.abs(self.state_vector)**2 # Probabilidades son magnitud al cuadrado
        logger.info(f"QuantumState inicializado con {num_positions} posiciones.")

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        """Normaliza un vector de estado para que tenga norma 1."""
        norm = np.linalg.norm(state)
        return state / norm if norm > 1e-9 else state # Evitar división por cero

    def predict_state_update(self) -> Tuple[np.ndarray, float]:
        """Predice el siguiente estado usando métodos heredados."""
        # Necesita al menos 2 dimensiones para Mahalanobis, usamos las primeras 2 componentes
        if self.num_positions < 2:
             # Fallback simple si no hay suficientes dimensiones
             logger.warning("Menos de 2 dimensiones, predicción de estado simplificada.")
             return self.state_vector, 0.5 # Estado actual, posterior neutro

        # Usar las primeras 2 componentes para cálculos que lo requieren
        state_2d = self.state_vector[:2].reshape(1, -1) if self.num_positions >= 2 else np.array([[0.0, 0.0]])
        # Calcular entropía sobre todo el vector de probabilidades
        entropy = shannon_entropy(self.probabilities) # Usa la función global/importada
        coherence = np.mean(self.probabilities) # Coherencia simple como media

        try:
             # Llama al método heredado que usa Mahalanobis, etc.
             # Nota: predict_quantum_state espera (n_muestras, 2), usamos state_2d
             next_state_prediction_scalar, posterior_scalar = self.predict_quantum_state(state_2d, entropy, coherence)
             # Reconstruir un vector de estado basado en la predicción escalar (simplificación)
             # Se podría distribuir la predicción o usarla para sesgar una nueva distribución.
             # Aquí, creamos un vector sesgado hacia la predicción (ej. con ruido gaussiano)
             predicted_vector = np.random.normal(loc=next_state_prediction_scalar.numpy(), scale=0.1, size=self.num_positions)
             next_state_normalized = self.normalize_state(predicted_vector)
             posterior_val = posterior_scalar.numpy()

        except Exception as e:
             logger.error(f"Error en predict_quantum_state: {e}. Usando estado actual.")
             next_state_normalized = self.state_vector
             posterior_val = 0.5 # Posterior neutro en caso de error

        return next_state_normalized, posterior_val

    def update_state(self, action: int) -> None:
        """Actualiza el estado cuántico simulado."""
        next_state, posterior = self.predict_state_update()

        # Modificar factor de aprendizaje basado en acción y posterior
        if action == 1:
            update_factor = self.learning_rate * (1 + posterior) # Aprende más del nuevo estado
        else:
            update_factor = self.learning_rate * (1 - posterior) # Conserva más el estado actual
        update_factor = np.clip(update_factor, 0.01, 0.99) # Limitar factor

        # Actualización ponderada
        updated_state = (1 - update_factor) * self.state_vector + update_factor * next_state
        self.state_vector = self.normalize_state(updated_state)
        self.probabilities = np.abs(self.state_vector)**2 # Actualizar probabilidades
        logger.debug(f"Estado actualizado con acción {action}, posterior {posterior:.4f}, factor {update_factor:.4f}")

    def compute_quantum_uncertainty(self) -> float:
        """Calcula incertidumbre (entropía) del estado."""
        return shannon_entropy(self.probabilities) # Usa la función global/importada

    def quantum_interference(self, other_state: 'QuantumState') -> np.ndarray:
        """Simula interferencia (basada en producto punto)."""
        if self.num_positions != other_state.num_positions:
            raise ValueError("Los estados deben tener la misma dimensión para interferir.")
        # Simulación simple de interferencia (puede ser más compleja)
        # Usamos el ángulo entre vectores como factor de interferencia
        dot_product = np.dot(self.state_vector, other_state.state_vector)
        # Asegurarse que el producto punto esté en [-1, 1] para arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        interference_angle = np.arccos(dot_product)
        # El nuevo estado es una rotación/modulación basada en el ángulo
        new_state = self.state_vector * np.cos(interference_angle) + other_state.state_vector * np.sin(interference_angle)
        return self.normalize_state(new_state)

    def quantum_entanglement_measure(self, other_state: 'QuantumState') -> float:
        """Calcula una medida de entrelazamiento (basada en entropía del producto tensorial)."""
        if self.num_positions != other_state.num_positions:
             raise ValueError("Los estados deben tener la misma dimensión.")
        # Producto tensorial de probabilidades (representa estado conjunto no entrelazado)
        joint_probs_flat = np.outer(self.probabilities, other_state.probabilities).flatten()
        # El entrelazamiento podría medirse comparando la entropía del estado real
        # con la suma de entropías individuales o la del producto tensorial.
        # Aquí usamos la entropía del producto tensorial como una métrica relacionada.
        # Nota: Una medida más rigurosa requeriría la matriz de densidad.
        entanglement_metric = shannon_entropy(joint_probs_flat)
        # Normalizar (aproximado, ya que max entropía depende de N*N)
        max_entropy = np.log2(self.num_positions**2)
        return entanglement_metric / max_entropy if max_entropy > 0 else 0.0


    def visualize_state(self) -> Dict[str, Any]:
        """Devuelve un diccionario con información del estado para visualización."""
        return {
            'state_vector': self.state_vector.tolist(),
            'probabilities': self.probabilities.tolist(),
            'uncertainty': self.compute_quantum_uncertainty(),
            'norm': np.linalg.norm(self.state_vector)
        }

    def quantum_measurement(self, observable: Optional[np.ndarray] = None) -> float:
        """Simula una medición proyectiva."""
        if observable is None:
            # Medición simple en base computacional: devuelve índice basado en probabilidades
            measured_index = np.random.choice(self.num_positions, p=self.probabilities)
            # Podríamos colapsar el estado aquí si quisiéramos simularlo post-medición
            # self.state_vector = np.zeros(self.num_positions)
            # self.state_vector[measured_index] = 1.0
            # self.probabilities = np.abs(self.state_vector)**2
            return float(measured_index) # Devolver el índice medido
        else:
             # Medición con un observable Hermitiano
             if observable.shape != (self.num_positions, self.num_positions):
                  raise ValueError("Observable debe ser una matriz cuadrada con la dimensión del estado.")
             # Calcular valor esperado <psi|O|psi>
             expectation_value = np.dot(self.state_vector.conj().T, np.dot(observable, self.state_vector))
             # El valor esperado puede ser complejo si O no es Hermitiano, pero debería ser real.
             return np.real(expectation_value)


    def serialize_state(self) -> Dict[str, Any]:
        """Serializa el estado cuántico simulado."""
        return {
            'state_vector': self.state_vector.tolist(),
            # 'probabilities': self.probabilities.tolist(), # Se puede recalcular desde state_vector
            'num_positions': self.num_positions,
            'learning_rate': self.learning_rate
        }

    @classmethod
    def deserialize_state(cls, serialized_data: Dict[str, Any]) -> 'QuantumState':
        """Deserializa y reconstruye un QuantumState."""
        state = cls(
            num_positions=serialized_data['num_positions'],
            learning_rate=serialized_data['learning_rate']
        )
        state.state_vector = cls.normalize_state(np.array(serialized_data['state_vector']))
        state.probabilities = np.abs(state.state_vector)**2 # Recalcular probabilidades
        return state


# --- Ejemplo de Uso ---
if __name__ == "__main__":
    logger.info("--- Iniciando Demostración Híbrida ---")

    # 1. Inicializar Red Neuronal Clásica
    nn = NeuralNetwork(input_size=10, hidden_sizes=[8, 5], output_size=2, activation="relu")
    logger.info(f"Red Neuronal Clásica creada con arquitectura: {[10] + nn.hidden_sizes + [2]}")

    # Datos de ejemplo para la red neuronal
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, size=(50, 2)) # Salida binaria ejemplo

    # Entrenar la red clásica (ej. 10 épocas)
    logger.info("Entrenando red neuronal clásica...")
    for epoch in range(10):
        nn.backward(X_train, y_train, learning_rate=0.01, optimizer="adam")
    logger.info("Entrenamiento clásico completado.")

    # 2. Inicializar Activador Cuántico-Clásico
    qc_activator = QuantumClassicalActivator(n_qubits=1)
    logger.info("Activador Cuántico-Clásico inicializado.")

    # Crear un circuito cuántico para Sigmoid (ejemplo)
    qc_sigmoid, theta_sigmoid = qc_activator.create_quantum_activation_circuit(ActivationType.SIGMOID)
    logger.info("Circuito cuántico para Sigmoid creado:")
    # print(qc_sigmoid) # Descomentar para ver el circuito Qiskit

    # 3. Inicializar Estado Cuántico Simulado
    q_state = QuantumState(num_positions=4, learning_rate=0.05) # Estado de 2 qubits simulado (4 posiciones)
    logger.info("Estado Cuántico Simulado (QuantumState) inicializado.")

    # 4. Simular un paso híbrido
    logger.info("--- Simulando Paso Híbrido ---")
    # Tomar una entrada de ejemplo
    input_sample = X_train[0:1, :] # Tomar la primera muestra (mantener 2D)

    # a) Forward pass clásico
    pre_acts_Z, acts_A = nn.forward(input_sample)
    logger.info(f"Salida clásica de la red (última capa): {acts_A[-1]}")

    # b) Aplicar activación híbrida (conceptual) a la salida de la penúltima capa
    pre_activation_last_hidden = pre_acts_Z[-2] # Pre-activación antes de la capa de salida
    # Usar el promedio de pre-activaciones como entrada escalar para la activación híbrida
    scalar_input_for_hybrid = float(np.mean(pre_activation_last_hidden))

    hybrid_activation_value = qc_activator.quantum_influenced_activation(
        scalar_input_for_hybrid,
        ActivationType.RELU, # Usar la activación de la red
        quantum_influence=0.2 # Poner algo de influencia cuántica simulada
    )
    logger.info(f"Valor de activación híbrida (conceptual): {hybrid_activation_value:.4f}")
    # Nota: Esta activación híbrida no se reinyecta directamente en este flujo simple,
    # pero ilustra cómo podría calcularse. Una integración real requeriría modificar
    # la capa de activación de la red.

    # c) Actualizar el QuantumState basado en alguna acción (ej. derivada de la salida de la red)
    # Decidir una acción basada en la salida de la red (ej. clase predicha)
    action_from_nn = int(np.argmax(acts_A[-1]))
    logger.info(f"Acción derivada de la salida de la red: {action_from_nn}")
    q_state.update_state(action=action_from_nn)
    logger.info("Estado Cuántico Simulado actualizado.")

    # d) Visualizar estado cuántico simulado
    q_state_info = q_state.visualize_state()
    logger.info(f"Estado Cuántico Simulado después de la actualización: {q_state_info}")

    # e) Medición simulada
    measurement_result = q_state.quantum_measurement()
    logger.info(f"Resultado de medición simulada (base computacional): {measurement_result}")

    logger.info("--- Fin de la Demostración ---")