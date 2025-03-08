import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List, Dict, Union, Any
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import functools
import time

def timer_decorator(func):
    """
    Decorador que mide el tiempo de ejecución de una función.
    
    Args:
        func: La función a medir.
    
    Returns:
        La función decorada que muestra el tiempo de ejecución.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Función {func.__name__} ejecutada en {end_time - start_time:.4f} segundos")
        return result
    return wrapper


def validate_input_decorator(min_val=0.0, max_val=1.0):
    """
    Decorador que valida que los argumentos numéricos estén en un rango específico.
    
    Args:
        min_val (float): Valor mínimo permitido.
        max_val (float): Valor máximo permitido.
    
    Returns:
        La función decorada con validación de rango.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validar args que sean floats
            for i, arg in enumerate(args[1:], 1):  # Ignorar self si es un método
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(f"Argumento {i} debe estar entre {min_val} y {max_val}. Valor recibido: {arg}")
            
            # Validar kwargs que sean floats
            for name, arg in kwargs.items():
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(f"Argumento {name} debe estar entre {min_val} y {max_val}. Valor recibido: {arg}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class BayesLogic:
    """
    Clase para calcular probabilidades y seleccionar acciones basadas en el teorema de Bayes.

    Provee métodos para:
      - Calcular la probabilidad posterior usando Bayes.
      - Calcular probabilidades condicionales.
      - Derivar probabilidades previas en función de la entropía y la coherencia.
      - Calcular probabilidades conjuntas a partir de la coherencia, acción e influencia.
      - Seleccionar la acción final según un umbral predefinido.
      
    Atributos:
        EPSILON (float): Valor pequeño para evitar divisiones por cero.
        HIGH_ENTROPY_THRESHOLD (float): Umbral para considerar entropía alta (> 0.8).
        HIGH_COHERENCE_THRESHOLD (float): Umbral para considerar coherencia alta (> 0.6).
        ACTION_THRESHOLD (float): Umbral para decidir una acción positiva (> 0.5).
    """
    # Declaramos las constantes como atributos de clase
    EPSILON = 1e-6
    HIGH_ENTROPY_THRESHOLD = 0.8  # Valores superiores indican alta incertidumbre en los datos
    HIGH_COHERENCE_THRESHOLD = 0.6  # Valores superiores indican alta consistencia en los datos
    ACTION_THRESHOLD = 0.5  # Umbral para decidir entre acciones (valores > 0.5 favorecen acción=1)

    def __init__(self):
        # No duplicamos la declaración de las constantes, utilizamos las de clase
        pass

    @validate_input_decorator(0.0, 1.0)
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
        
        Raises:
            ValueError: Si alguna probabilidad está fuera del rango [0,1].
        """
        # Evitamos división por cero
        prior_b = prior_b if prior_b != 0 else self.EPSILON
        return (conditional_b_given_a * prior_a) / prior_b

    @validate_input_decorator(0.0, 1.0)
    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """
        Calcula la probabilidad condicional a partir de una probabilidad conjunta y una probabilidad previa.
        
        Args:
            joint_probability (float): Probabilidad conjunta de dos eventos.
            prior (float): Probabilidad previa del evento.
        
        Returns:
            float: La probabilidad condicional resultante.
        
        Raises:
            ValueError: Si alguna probabilidad está fuera del rango [0,1].
        """
        prior = prior if prior != 0 else self.EPSILON
        return joint_probability / prior

    @validate_input_decorator(0.0, 1.0)
    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """
        Deriva una probabilidad previa en función del valor de entropía.
        
        Lógica:
        - Si la entropía supera el umbral (HIGH_ENTROPY_THRESHOLD = 0.8), asignamos
          una probabilidad previa mayor (0.3) porque hay más incertidumbre.
        - Si la entropía es baja, asignamos una probabilidad previa menor (0.1)
          porque hay menos incertidumbre en los datos.
        
        Args:
            entropy (float): Valor de entropía entre 0 y 1.
        
        Returns:
            float: Retorna 0.3 si la entropía supera el umbral, o 0.1 en otro caso.
        
        Raises:
            ValueError: Si entropy está fuera del rango [0,1].
        """
        return 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.1

    @validate_input_decorator(0.0, 1.0)
    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """
        Deriva una probabilidad previa en función del valor de coherencia.
        
        Lógica:
        - Si la coherencia supera el umbral (HIGH_COHERENCE_THRESHOLD = 0.6), asignamos
          una probabilidad previa mayor (0.6) ya que los datos son más consistentes.
        - Si la coherencia es baja, asignamos una probabilidad previa menor (0.2)
          indicando menor confianza en la consistencia de los datos.
        
        Args:
            coherence (float): Valor de coherencia entre 0 y 1.
        
        Returns:
            float: Retorna 0.6 si la coherencia supera el umbral, o 0.2 en otro caso.
        
        Raises:
            ValueError: Si coherence está fuera del rango [0,1].
        """
        return 0.6 if coherence > self.HIGH_COHERENCE_THRESHOLD else 0.2

    @validate_input_decorator(0.0, 1.0)
    def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
        """
        Calcula la probabilidad conjunta de A y B basándose en la coherencia, la acción e influencia PRN.
        
        Lógica:
        - Si la coherencia es alta (> HIGH_COHERENCE_THRESHOLD):
          * Para action=1: La probabilidad conjunta se calcula como una combinación ponderada
            que favorece valores altos cuando prn_influence es alta.
          * Para action=0: La probabilidad conjunta se calcula como una combinación ponderada
            que favorece valores bajos cuando prn_influence es alta.
        - Si la coherencia es baja: Se utiliza un valor fijo (0.3) independiente de otros factores.
        
        Args:
            coherence (float): Valor de coherencia entre 0 y 1.
            action (int): Indicador de acción (1 para acción positiva, 0 para negativa).
            prn_influence (float): Factor de influencia PRN entre 0 y 1.
        
        Returns:
            float: Probabilidad conjunta resultante entre 0 y 1.
        
        Raises:
            ValueError: Si coherence o prn_influence están fuera del rango [0,1].
        """
        if coherence > self.HIGH_COHERENCE_THRESHOLD:
            if action == 1:
                # Si la acción es positiva, la influencia alta del PRN aumenta la probabilidad
                return prn_influence * 0.8 + (1 - prn_influence) * 0.2
            else:
                # Si la acción es negativa, la influencia alta del PRN disminuye la probabilidad
                return prn_influence * 0.1 + (1 - prn_influence) * 0.7
        # Para coherencia baja, usamos un valor intermedio fijo
        return 0.3

    @timer_decorator
    @validate_input_decorator(0.0, 1.0)
    def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, 
                                                prn_influence: float, action: int) -> Dict[str, float]:
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
            entropy (float): Valor de entropía entre 0 y 1.
            coherence (float): Valor de coherencia entre 0 y 1.
            prn_influence (float): Factor de influencia PRN entre 0 y 1.
            action (int): Indicador de acción (1 o 0).
        
        Returns:
            dict: Resultados con:
                - "action_to_take": Acción seleccionada.
                - "high_entropy_prior": Probabilidad previa basada en entropía.
                - "high_coherence_prior": Probabilidad previa basada en coherencia.
                - "posterior_a_given_b": Probabilidad posterior.
                - "conditional_action_given_b": Probabilidad condicional para la acción.
        
        Raises:
            ValueError: Si los parámetros de probabilidad están fuera del rango [0,1].
        """
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        # La probabilidad condicional se ajusta según el nivel de entropía
        conditional_b_given_a = (prn_influence * 0.7 + (1 - prn_influence) * 0.3
                                if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.2)
        
        # Cálculo del posterior usando el teorema de Bayes
        posterior_a_given_b = self.calculate_posterior_probability(
            high_entropy_prior, high_coherence_prior, conditional_b_given_a
        )
        
        # Probabilidad conjunta basada en coherencia, acción e influencia PRN
        joint_probability_ab = self.calculate_joint_probability(
            coherence, action, prn_influence
        )
        
        # Probabilidad condicional de la acción dado B
        conditional_action_given_b = self.calculate_conditional_probability(
            joint_probability_ab, high_coherence_prior
        )
        
        # Decisión final basada en el umbral de acción
        action_to_take = 1 if conditional_action_given_b > self.ACTION_THRESHOLD else 0

        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }


# Convertimos la clase StatisticalAnalysis a funciones independientes
def shannon_entropy(data: List[Any]) -> float:
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


def calculate_covariance_between_two_variables(data: tf.Tensor) -> Tuple[float, float]:
    """
    Calcula la covarianza entre dos variables, tanto de forma manual como utilizando 
    TensorFlow Probability.
    
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


@timer_decorator
def compute_mahalanobis_distance(data: List[List[float]], point: List[float]) -> float:
    """
    Calcula la distancia de Mahalanobis entre un punto y un conjunto de datos.

    La distancia de Mahalanobis es una medida de la distancia entre un punto y una distribución, 
    teniendo en cuenta la correlación entre las variables. Es útil para detectar outliers
    en un espacio multidimensional.

    Args:
        data (list o array-like): Conjunto de datos, donde cada fila es una observación.
        point (list o array-like): Punto para el cual se calcula la distancia.
    
    Returns:
        float: La distancia de Mahalanobis.
    
    Raises:
        ValueError: Si la matriz de covarianza no es invertible.
    """
    data_array = np.array(data)
    point_array = np.array(point)

    # Estimar la matriz de covarianza empírica
    covariance_estimator = EmpiricalCovariance().fit(data_array)
    cov_matrix = covariance_estimator.covariance_
    print("Matriz de covarianza:")
    print(cov_matrix)

    try:
        # Intentamos calcular la inversa de la matriz de covarianza
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Si no es invertible, usamos la pseudo-inversa
        print("ADVERTENCIA: Matriz de covarianza singular, usando pseudo-inversa")
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # Calculamos el vector de medias
    mean_vector = np.mean(data_array, axis=0)
    
    # Calculamos la distancia de Mahalanobis
    distance = mahalanobis(point_array, mean_vector, inv_cov_matrix)
    return distance


class PRN:
    """
    Clase para modelar el Ruido Probabilístico de Referencia (Probabilistic Reference Noise).
    
    Esta clase generalizada puede ser utilizada para representar cualquier tipo de
    influencia probabilística en un sistema.
    
    Atributos:
        influence (float): Factor de influencia entre 0 y 1.
        parameters (dict): Parámetros adicionales específicos del algoritmo.
    """
    def __init__(self, influence: float, algorithm_type: str = None, **parameters):
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
    
    def __str__(self) -> str:
        """
        Representación en string del objeto PRN.
        
        Returns:
            str: Descripción del objeto PRN.
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"PRN(influence={self.influence:.4f}{algo_str}{', ' + params_str if params_str else ''})"


# --- Ejemplo de uso refactorizado ---
def run_bayes_logic_example():
    """Ejecuta un ejemplo de uso de BayesLogic con valores predefinidos."""
    # Calcular entropía desde datos de ejemplo
    entropy_value = shannon_entropy([1, 2, 3, 4, 5, 5, 2])
    coherence_value = 0.7
    prn_influence = 0.8
    action_input = 1

    bayes = BayesLogic()
    decision = bayes.calculate_probabilities_and_select_action(
        entropy_value, coherence_value, prn_influence, action_input
    )
    
    print("\n=== Decisión Bayesiana ===")
    for key, value in decision.items():
        print(f"  {key}: {value}")


def run_statistical_analysis_example():
    """Ejecuta ejemplos de los diferentes análisis estadísticos."""
    # Datos para los ejemplos
    entropy_value = shannon_entropy([1, 2, 3, 4, 5, 5, 2])
    prn_influence = 0.8
    
    # 1. Cálculo de cosenos direccionales
    cos_x, cos_y, cos_z = calculate_cosines(entropy_value, prn_influence)
    print("\n=== Cosenos direccionales ===")
    print(f"Entropía: {entropy_value:.4f}")
    print(f"cos_x = {cos_x:.4f}, cos_y = {cos_y:.4f}, cos_z = {cos_z:.4f}")

    # 2. Cálculo de la matriz de covarianza para múltiples variables
    datos_multivariados = tf.constant([
        [2.1, 8.0, -1.0],
        [2.5, 10.0, 0.5],
        [3.6, 12.0, 1.2],
        [4.0, 14.0, 2.3],
        [5.1, 16.0, 3.4]
    ], dtype=tf.float32)
    
    cov_matrix = calculate_covariance_matrix(datos_multivariados)
    print("\n=== Matriz de Covarianza (multivariada) ===")
    print(cov_matrix)

    # 3. Cálculo de la covarianza entre dos variables
    datos_dos_vars = tf.constant([
        [2.1, 8.0],
        [2.5, 10.0],
        [3.6, 12.0],
        [4.0, 14.0],
        [5.1, 16.0]
    ], dtype=tf.float32)
    
    cov_manual, cov_tfp = calculate_covariance_between_two_variables(datos_dos_vars)
    print("\n=== Covarianza entre dos variables ===")
    print(f"Covarianza (Manual): {cov_manual:.4f}")
    print(f"Covarianza (TFP): {cov_tfp:.4f}")

    # 4. Cálculo de la distancia de Mahalanobis
    point = [2, 4]
    data = [[1, 2], [3, 5], [5, 6]]
    distance = compute_mahalanobis_distance(data, point)
    print(f"\nDistancia de Mahalanobis: {distance:.4f}")

    # 5. Datos de ejemplo nq_a y nq_b
    nq_a = np.array([[0.8, 0.2], [0.9, 0.4], [0.1, 0.7]])
    nq_b = np.array([[0.78, 0.22], [0.88, 0.35], [0.12, 0.68]])
    print("\n=== Datos de ejemplo (nq_a y nq_b) ===")
    print("nq_a:")
    print(nq_a)
    print("nq_b:")
    print(nq_b)


def run_prn_example():
    """Ejecuta un ejemplo de uso de la clase PRN generalizada."""
    # Crear un objeto PRN con algoritmo personalizado
    prn = PRN(
        influence=0.65, 
        algorithm_type="bayesian", 
        decay_rate=0.05, 
        min_threshold=0.2
    )
    
    # Ajustar la influencia
    prn.adjust_influence(-0.15)
    
    # Crear otro PRN para combinar
    prn2 = PRN(influence=0.8, algorithm_type="monte_carlo", iterations=1000)
    
    # Combinar los PRN
    combined_prn = prn.combine_with(prn2, weight=0.7)
    
    print("\n=== Ejemplo de PRN ===")
    print(f"PRN original: {prn}")
    print(f"PRN secundario: {prn2}")
    print(f"PRN combinado: {combined_prn}")
    
    # Usar el PRN con BayesLogic
    bayes = BayesLogic()
    entropy = 0.9
    coherence = 0.7
    action = 1
    
    decision = bayes.calculate_probabilities_and_select_action(
        entropy, coherence, combined_prn.influence, action
    )
    
    print("\n=== Decisión con PRN combinado ===")
    for key, value in decision.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("===== DEMOSTRACION DE FUNCIONALIDADES =====")
    
    # Ejecutar ejemplos individuales
    run_bayes_logic_example()
    run_statistical_analysis_example()
    run_prn_example()
    
    # Ejemplo de integración final
    print("\n===== EJEMPLO DE INTEGRACIÓN FINAL =====")
    
    # Crear un objeto PRN personalizado
    prn = PRN(influence=0.5, algorithm_type="bayesian_inference")
    
    # Definir valores de entropía y coherencia
    entropy = 0.9
    coherence = 0.1
    previous_action = 1
    
    # Instanciar BayesLogic
    bayes_logic = BayesLogic()
    
    # Calcular la acción a tomar
    result = bayes_logic.calculate_probabilities_and_select_action(
        entropy, coherence, prn.influence, previous_action
    )
    
    # Determinar la acción final
    action_to_take = result["action_to_take"]
    
    print(f"Acción a tomar: {'Move Right' if action_to_take == 1 else 'Move Left'}")
    print(f"Probabilidad condicional: {result['conditional_action_given_b']:.4f}")
    print(f"Umbral para decisión: {BayesLogic.ACTION_THRESHOLD}")