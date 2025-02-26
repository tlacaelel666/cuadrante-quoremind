import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance

class BayesLogic:
    """
    Clase para calcular probabilidades y seleccionar acciones basadas en el teorema de Bayes.

    Provee métodos para:
      - Calcular la probabilidad posterior usando Bayes.
      - Calcular probabilidades condicionales.
      - Derivar probabilidades previas en función de la entropía y la coherencia.
      - Calcular probabilidades conjuntas a partir de la coherencia, acción e influencia PRN.
      - Seleccionar la acción final según un umbral predefinido.
    """
    EPSILON = 1e-6
    HIGH_ENTROPY_THRESHOLD = 0.8
    HIGH_COHERENCE_THRESHOLD = 0.6
    ACTION_THRESHOLD = 0.5

    def __init__(self):
        self.EPSILON = 1e-6
        self.HIGH_ENTROPY_THRESHOLD = 0.8
        self.HIGH_COHERENCE_THRESHOLD = 0.6
        self.ACTION_THRESHOLD = 0.5

    def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
        """
        Calcula la probabilidad posterior usando el teorema de Bayes:
            P(A|B) = (P(B|A) * P(A)) / P(B)
        
        Args:
            prior_a (float): Probabilidad previa de A.
            prior_b (float): Probabilidad previa de B.
            conditional_b_given_a (float): Probabilidad condicional de B dado A.
        
        Returns:
            float: La probabilidad posterior P(A|B).
        """
        prior_b = prior_b if prior_b != 0 else self.EPSILON
        return (conditional_b_given_a * prior_a) / prior_b

    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """
        Calcula la probabilidad condicional a partir de una probabilidad conjunta y una probabilidad previa.
        
        Args:
            joint_probability (float): Probabilidad conjunta de dos eventos.
            prior (float): Probabilidad previa del evento.
        
        Returns:
            float: La probabilidad condicional resultante.
        """
        prior = prior if prior != 0 else self.EPSILON
        return joint_probability / prior

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """
        Deriva una probabilidad previa en función del valor de entropía.
        
        Args:
            entropy (float): Valor de entropía.
        
        Returns:
            float: Retorna 0.3 si la entropía supera el umbral, o 0.1 en otro caso.
        """
        return 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.1

    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """
        Deriva una probabilidad previa en función del valor de coherencia.
        
        Args:
            coherence (float): Valor de coherencia.
        
        Returns:
            float: Retorna 0.6 si la coherencia supera el umbral, o 0.2 en otro caso.
        """
        return 0.6 if coherence > self.HIGH_COHERENCE_THRESHOLD else 0.2

    def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
        """
        Calcula la probabilidad conjunta de A y B basándose en la coherencia, la acción e influencia PRN(Probabilistic Record Noise).
        
        Si la coherencia es mayor que el umbral, se aplican ponderaciones distintas según el valor de 'action'.
        En caso contrario, se retorna un valor fijo de 0.3.
        
        Args:
            coherence (float): Valor de coherencia.
            action (int): Indicador de acción (1 para acción positiva, 0 para negativa).
            prn_influence (float): Factor de influencia PRN.
        
        Returns:
            float: Probabilidad conjunta resultante.
        """
        if coherence > self.HIGH_COHERENCE_THRESHOLD:
            if action == 1:
                return prn_influence * 0.8 + (1 - prn_influence) * 0.2
            else:
                return prn_influence * 0.1 + (1 - prn_influence) * 0.7
        return 0.3

    def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, prn_influence: float, action: int) -> dict:
        """
        Integra los cálculos bayesianos para determinar la acción a tomar basándose en entropía, coherencia,
        influencia PRN y un indicador de acción.
        
        Flujo:
          1. Calcular probabilidades previas a partir de entropía y coherencia.
          2. Determinar la probabilidad condicional (ajustada por el valor de entropía).
          3. Calcular la probabilidad posterior.
          4. Calcular la probabilidad conjunta.
          5. Derivar la probabilidad condicional para la acción.
          6. Seleccionar la acción final si la probabilidad supera un umbral definido.
        
        Args:
            entropy (float): Valor de entropía.
            coherence (float): Valor de coherencia.
            prn_influence (float): Factor de influencia PRN.
            action (int): Indicador de acción (1 o 0).
        
        Returns:
            dict: Resultados con:
                - "action_to_take": Acción seleccionada.
                - "high_entropy_prior": Probabilidad previa basada en entropía.
                - "high_coherence_prior": Probabilidad previa basada en coherencia.
                - "posterior_a_given_b": Probabilidad posterior.
                - "conditional_action_given_b": Probabilidad condicional para la acción.
        """
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


class StatisticalAnalysis:
    """
    Clase para realizar análisis estadísticos y cálculos adicionales:
      - Cálculo de la entropía de Shannon.
      - Cálculo de cosenos direccionales.
      - Cálculo de la matriz de covarianza y covarianza entre dos variables usando TensorFlow.
      - Cálculo de la distancia de Mahalanobis.
    """

    @staticmethod
    def shannon_entropy(data: list) -> float:
        """
        Calcula la entropía de Shannon para un conjunto de datos.

        Args:
            data (list o numpy.ndarray): Datos de entrada.

        Returns:
            float: Entropía en bits.
        """
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        probabilities = probabilities[probabilities > 0]  # Evita log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def calculate_cosines(entropy: float, prn_object: float) -> Tuple[float, float, float]:
        """
        Calcula los cosenos direccionales (x, y, z) para un vector 3D.

        Args:
            entropy (float): Valor de entropía (componente x).
            prn_object (float): Valor del ambiente de ruido (componente y).

        Returns:
            tuple: (cos_x, cos_y, cos_z) correspondientes a cada dirección.
        """
        # Evitar división por cero
        if entropy == 0:
            entropy = 1e-6
        if prn_object == 0:
            prn_object = 1e-6

        magnitude = np.sqrt(entropy ** 2 + prn_object ** 2 + 1)
        cos_x = entropy / magnitude
        cos_y = prn_object / magnitude
        cos_z = 1 / magnitude
        return cos_x, cos_y, cos_z

    @staticmethod
    def calculate_covariance_matrix(data: tf.Tensor) -> np.ndarray:
        """
        Calcula la matriz de covarianza de un conjunto de datos usando TensorFlow Probability.
        
        Se asume que cada fila es una observación y cada columna una variable.
        
        Args:
            data (tf.Tensor): Tensor de datos (dtype tf.float32).
        
        Returns:
            np.ndarray: Matriz de covarianza evaluada como un array de NumPy.
        """
        cov_matrix = tfp.stats.covariance(data, sample_axis=0, event_axis=None)
        return cov_matrix.numpy()

    @staticmethod
    def calculate_covariance_between_two_variables(data: tf.Tensor) -> Tuple[float, float]:
        """
        Calcula la covarianza entre dos variables, tanto de forma manual como utilizando TensorFlow Probability.
        
        Se asume que 'data' tiene dos columnas (cada columna representa una variable).
        
        Args:
            data (tf.Tensor): Tensor de datos de dos variables (dtype tf.float32).
        
        Returns:
            tuple: (cov_manual, cov_tfp) donde:
                - cov_manual: Covarianza calculada manualmente.
                - cov_tfp: Covarianza calculada con tfp.stats.covariance.
        """
        x = data[:, 0]
        y = data[:, 1]
        media_x = tf.reduce_mean(x)
        media_y = tf.reduce_mean(y)
        n = tf.cast(tf.size(x), tf.float32)
        cov_manual = tf.reduce_mean((x - media_x) * (y - media_y)) * ((n - 1) / n)
        cov_tfp = tfp.stats.covariance(data, sample_axis=0, event_axis=None)[0, 1]
        return cov_manual.numpy(), cov_tfp.numpy()

    @staticmethod
    def compute_mahalanobis_distance(data: list, point: list) -> float:
        """
        Calcula la distancia de Mahalanobis entre un punto y un conjunto de datos.

        La distancia de Mahalanobis es una medida de la distancia entre un punto y una distribución, 
        teniendo en cuenta la correlación entre las variables.

        Args:
            data (list o array-like): Conjunto de datos, donde cada fila es una observación.
            point (list o array-like): Punto para el cual se calcula la distancia.
        
        Returns:
            float: La distancia de Mahalanobis.
        """
        data_array = np.array(data)
        point_array = np.array(point)

        # Estimar la matriz de covarianza empírica
        covariance_estimator = EmpiricalCovariance().fit(data_array)
        cov_matrix = covariance_estimator.covariance_
        print("Covariance matrix:", cov_matrix)

        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)

        mean_vector = np.mean(data_array, axis=0)
        distance = mahalanobis(point_array, mean_vector, inv_cov_matrix)
        return distance


# Ejemplo de uso integrado:
if __name__ == "__main__":
    # --- Ejemplo para BayesLogic ---
    entropy_value = StatisticalAnalysis.shannon_entropy([1, 2, 3, 4, 5, 5, 2])
    coherence_value = 0.7
    prn_influence = 0.8  # Este valor representa la influencia del ambiente (ahora referenciado como prn_object en otros métodos)
    action_input = 1

    bayes = BayesLogic()
    decision = bayes.calculate_probabilities_and_select_action(entropy_value, coherence_value, prn_influence, action_input)
    print("Decisión Bayesiana:")
    for key, value in decision.items():
        print(f"  {key}: {value}")

    # --- Ejemplo para análisis estadístico ---
    # 1. Cálculo de cosenos direccionales
    cos_x, cos_y, cos_z = StatisticalAnalysis.calculate_cosines(entropy_value, prn_influence)
    print(f"\nEntropía: {entropy_value:.4f}")
    print(f"Cosenos direccionales: cos_x = {cos_x:.4f}, cos_y = {cos_y:.4f}, cos_z = {cos_z:.4f}")

    # 2. Cálculo de la matriz de covarianza para múltiples variables
    datos_multivariados = tf.constant([
        [2.1, 8.0, -1.0],
        [2.5, 10.0, 0.5],
        [3.6, 12.0, 1.2],
        [4.0, 14.0, 2.3],
        [5.1, 16.0, 3.4]
    ], dtype=tf.float32)
    cov_matrix = StatisticalAnalysis.calculate_covariance_matrix(datos_multivariados)
    print("\nMatriz de Covarianza (multivariada):")
    print(cov_matrix)

    # 3. Cálculo de la covarianza entre dos variables
    datos_dos_vars = tf.constant([
        [2.1, 8.0],
        [2.5, 10.0],
        [3.6, 12.0],
        [4.0, 14.0],
        [5.1, 16.0]
    ], dtype=tf.float32)
    cov_manual, cov_tfp = StatisticalAnalysis.calculate_covariance_between_two_variables(datos_dos_vars)
    print("\nCovarianza entre dos variables:")
    print(f"  Covarianza (Manual): {cov_manual:.4f}")
    print(f"  Covarianza (TFP): {cov_tfp:.4f}")

    # 4. Cálculo de la distancia de Mahalanobis
    point = [2, 4]
    data = [[1, 2], [3, 5], [5, 6]]  # Datos de ejemplo
    distance = StatisticalAnalysis.compute_mahalanobis_distance(data, point)
    print("\nMahalanobis distance:", distance)

    # 5. Ejemplo con conjuntos de datos nq_a y nq_b (puedes integrarlos según la lógica necesaria)
    nq_a = np.array([[0.8, 0.2], [0.9, 0.4], [0.1, 0.7]])
    nq_b = np.array([[0.78, 0.22], [0.88, 0.35], [0.12, 0.68]])
    print("\nDatos de ejemplo (nq_a y nq_b):")
    print("nq_a:")
    print(nq_a)
    print("nq_b:")
    print(nq_b)
import numpy as np
import torch

# Definición de la clase PRN (Probabilistic Reference Noise)
class PRN:
    def __init__(self, influence: float):
        self.influence = influence

# Instanciar la clase BayesLogic
bayes_logic = BayesLogic()

# Crear un objeto PRN con cierta influencia
prn = PRN(influence=0.5)

# Definir valores de entropía y coherencia
entropy = 0.9
coherence = 0.1

# Acción previa (puede ser 0 o 1)
previous_action = 1

# Calcular la acción a tomar basada en las probabilidades, utilizando prn.influence
result = bayes_logic.calculate_probabilities_and_select_action(
    entropy, coherence, prn.influence, previous_action
)
action_to_take = result["action_to_take"]

print(f"Acción a tomar: {'Move Right' if action_to_take == 1 else 'Move Left'}")
