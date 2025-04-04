import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List, Dict, Union, Any, Optional, Callable
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import functools
import time
from dataclasses import dataclass


def timer_decorator(func: Callable) -> Callable:
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


def validate_input_decorator(min_val: float = 0.0, max_val: float = 1.0) -> Callable:
    """
    Decorador que valida que los argumentos numéricos estén en un rango específico.
    
    Args:
        min_val: Valor mínimo permitido.
        max_val: Valor máximo permitido.
    
    Returns:
        La función decorada con validación de rango.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validar args que sean floats (ignorar self si es un método)
            for i, arg in enumerate(args[1:], 1):
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(f"Argumento {i} debe estar entre {min_val} y {max_val}. Valor recibido: {arg}")

            # Validar kwargs que sean floats
            for name, arg in kwargs.items():
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(f"Argumento {name} debe estar entre {min_val} y {max_val}. Valor recibido: {arg}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


@dataclass
class BayesLogicConfig:
    """Configuración para la clase BayesLogic."""
    epsilon: float = 1e-6
    high_entropy_threshold: float = 0.8
    high_coherence_threshold: float = 0.6
    action_threshold: float = 0.5


class BayesLogic:
    """
    Clase para calcular probabilidades y seleccionar acciones basadas en el teorema de Bayes.

    Provee métodos para:
      - Calcular la probabilidad posterior usando Bayes.
      - Calcular probabilidades condicionales.
      - Derivar probabilidades previas en función de la entropía y la coherencia.
      - Calcular probabilidades conjuntas a partir de la coherencia, acción e influencia.
      - Seleccionar la acción final según un umbral predefinido.
    """
    def __init__(self, config: Optional[BayesLogicConfig] = None):
        """
        Inicializa una instancia de BayesLogic con la configuración proporcionada.
        
        Args:
            config: Configuración opcional para la clase BayesLogic.
                   Si no se proporciona, se usa la configuración por defecto.
        """
        self.config = config or BayesLogicConfig()

    @validate_input_decorator(0.0, 1.0)
    def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
        """
        Calcula la probabilidad posterior usando el teorema de Bayes:
            P(A|B) = (P(B|A) * P(A)) / P(B)
        
        Args:
            prior_a: Probabilidad previa de A.
            prior_b: Probabilidad previa de B.
            conditional_b_given_a: Probabilidad condicional de B dado A.
        
        Returns:
            La probabilidad posterior P(A|B).
        """
        # Evitamos división por cero
        prior_b = max(prior_b, self.config.epsilon)
        return (conditional_b_given_a * prior_a) / prior_b

    @validate_input_decorator(0.0, 1.0)
    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """
        Calcula la probabilidad condicional a partir de una probabilidad conjunta y una probabilidad previa.
        
        Args:
            joint_probability: Probabilidad conjunta de dos eventos.
            prior: Probabilidad previa del evento.
        
        Returns:
            La probabilidad condicional resultante.
        """
        prior = max(prior, self.config.epsilon)
        return joint_probability / prior

    @validate_input_decorator(0.0, 1.0)
    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """
        Deriva una probabilidad previa en función del valor de entropía.
        
        Args:
            entropy: Valor de entropía entre 0 y 1.
        
        Returns:
            Retorna 0.3 si la entropía supera el umbral, o 0.1 en otro caso.
        """
        return 0.3 if entropy > self.config.high_entropy_threshold else 0.1

    @validate_input_decorator(0.0, 1.0)
    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """
        Deriva una probabilidad previa en función del valor de coherencia.
        
        Args:
            coherence: Valor de coherencia entre 0 y 1.
        
        Returns:
            Retorna 0.6 si la coherencia supera el umbral, o 0.2 en otro caso.
        """
        return 0.6 if coherence > self.config.high_coherence_threshold else 0.2

    @validate_input_decorator(0.0, 1.0)
    def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
        """
        Calcula la probabilidad conjunta de A y B basándose en la coherencia, la acción e influencia PRN.
        
        Args:
            coherence: Valor de coherencia entre 0 y 1.
            action: Indicador de acción (1 para acción positiva, 0 para negativa).
            prn_influence: Factor de influencia PRN entre 0 y 1.
        
        Returns:
            Probabilidad conjunta resultante entre 0 y 1.
        """
        if coherence > self.config.high_coherence_threshold:
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
    def calculate_probabilities_and_select_action(
        self, entropy: float, coherence: float, prn_influence: float, action: int
    ) -> Dict[str, float]:
        """
        Integra los cálculos bayesianos para determinar la acción a tomar.
        
        Args:
            entropy: Valor de entropía entre 0 y 1.
            coherence: Valor de coherencia entre 0 y 1.
            prn_influence: Factor de influencia PRN entre 0 y 1.
            action: Indicador de acción (1 o 0).
        
        Returns:
            Diccionario con los resultados del análisis y la acción a tomar.
        """
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        # La probabilidad condicional se ajusta según el nivel de entropía
        conditional_b_given_a = (
            prn_influence * 0.7 + (1 - prn_influence) * 0.3
            if entropy > self.config.high_entropy_threshold else 0.2
        )

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
        action_to_take = 1 if conditional_action_given_b > self.config.action_threshold else 0

        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }


class StatisticalAnalysis:
    """Clase que agrupa funciones de análisis estadístico."""
    
    @staticmethod
    def shannon_entropy(data: List[Any]) -> float:
        """
        Calcula la entropía de Shannon para un conjunto de datos.

        Args:
            data: Datos de entrada.

        Returns:
            Entropía en bits.
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
            entropy: Valor de entropía (componente x).
            prn_object: Valor del ambiente de ruido (componente y).

        Returns:
            (cos_x, cos_y, cos_z) correspondientes a cada dirección.
        """
        # Evitar división por cero
        epsilon = 1e-6
        entropy = max(entropy, epsilon)
        prn_object = max(prn_object, epsilon)

        magnitude = np.sqrt(entropy ** 2 + prn_object ** 2 + 1)
        cos_x = entropy / magnitude
        cos_y = prn_object / magnitude
        cos_z = 1 / magnitude
        return cos_x, cos_y, cos_z

    @staticmethod
    def calculate_covariance_matrix(data: tf.Tensor) -> np.ndarray:
        """
        Calcula la matriz de covarianza de un conjunto de datos usando TensorFlow Probability.
        
        Args:
            data: Tensor de datos (dtype tf.float32).
        
        Returns:
            Matriz de covarianza evaluada como un array de NumPy.
        """
        cov_matrix = tfp.stats.covariance(data, sample_axis=0, event_axis=None)
        return cov_matrix.numpy()

    @staticmethod
    def calculate_covariance_between_two_variables(data: tf.Tensor) -> Tuple[float, float]:
        """
        Calcula la covarianza entre dos variables.
        
        Args:
            data: Tensor de datos de dos variables (dtype tf.float32).
        
        Returns:
            (cov_manual, cov_tfp) calculados de dos formas diferentes.
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
    @timer_decorator
    def compute_mahalanobis_distance(data: List[List[float]], point: List[float]) -> float:
        """
        Calcula la distancia de Mahalanobis entre un punto y un conjunto de datos.

        Args:
            data: Conjunto de datos, donde cada fila es una observación.
            point: Punto para el cual se calcula la distancia.
        
        Returns:
            La distancia de Mahalanobis.
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


class ComplexPRN:
    """
    Clase PRN modificada para representar números complejos.
    """
    def __init__(self, real_component: float, imaginary_component: float, 
                 algorithm_type: Optional[str] = None, **parameters):
        """
        Inicializa un PRN complejo.
        
        Args:
            real_component: Componente real del número complejo.
            imaginary_component: Componente imaginaria del número complejo.
            algorithm_type: Tipo de algoritmo a utilizar.
            **parameters: Parámetros adicionales específicos del algoritmo.
        """
        self.real_component = real_component
        self.imaginary_component = imaginary_component
        # Módulo del número complejo como influencia
        self.influence = np.sqrt(real_component**2 + imaginary_component**2)
        self.algorithm_type = algorithm_type
        self.parameters = parameters

    def __str__(self) -> str:
        """Representación en string del objeto PRN complejo."""
        return (f"ComplexPRN(real={self.real_component:.4f}, "
                f"imag={self.imaginary_component:.4f}, "
                f"influence={self.influence:.4f})")


class PRN:
    """
    Clase para modelar el Ruido Probabilístico de Referencia (Probabilistic Reference Noise).
    """
    def __init__(self, influence: float, algorithm_type: Optional[str] = None, **parameters):
        """
        Inicializa un objeto PRN con un factor de influencia y parámetros específicos.
        
        Args:
            influence: Factor de influencia entre 0 y 1.
            algorithm_type: Tipo de algoritmo a utilizar.
            **parameters: Parámetros adicionales específicos del algoritmo.
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
            adjustment: Valor de ajuste (positivo o negativo).
        """
        new_influence = self.influence + adjustment
        # Truncamos al rango válido
        new_influence = max(0, min(1, new_influence))
        
        if new_influence != self.influence + adjustment:
            print(f"ADVERTENCIA: Influencia ajustada a {new_influence} para mantenerla en el rango [0,1]")
            
        self.influence = new_influence

    def combine_with(self, other_prn: 'PRN', weight: float = 0.5) -> 'PRN':
        """
        Combina este PRN con otro según un peso específico.
        
        Args:
            other_prn: Otro objeto PRN para combinar.
            weight: Peso para la combinación, entre 0 y 1 (por defecto 0.5).
        
        Returns:
            Un nuevo objeto PRN con la influencia combinada.
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
        """Representación en string del objeto PRN."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"PRN(influence={self.influence:.4f}{algo_str}{', ' + params_str if params_str else ''})"


# --- Funciones de demostración ---
def run_bayes_logic_example():
    """Ejecuta un ejemplo de uso de BayesLogic con valores predefinidos."""
    # Calcular entropía desde datos de ejemplo
    entropy_value = StatisticalAnalysis.shannon_entropy([1, 2, 3, 4, 5, 5, 2])
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
    stats = StatisticalAnalysis()
    
    # Datos para los ejemplos
    entropy_value = stats.shannon_entropy([1, 2, 3, 4, 5, 5, 2])
    prn_influence = 0.8
    
    # 1. Cálculo de cosenos direccionales
    cos_x, cos_y, cos_z = stats.calculate_cosines(entropy_value, prn_influence)
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
    
    cov_matrix = stats.calculate_covariance_matrix(datos_multivariados)
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
    
    cov_manual, cov_tfp = stats.calculate_covariance_between_two_variables(datos_dos_vars)
    print("\n=== Covarianza entre dos variables ===")
    print(f"Covarianza (Manual): {cov_manual:.4f}")
    print(f"Covarianza (TFP): {cov_tfp:.4f}")

    # 4. Cálculo de la distancia de Mahalanobis
    point = [2, 4]
    data = [[1, 2], [3, 5], [5, 6]]
    distance = stats.compute_mahalanobis_distance(data, point)
    print(f"\nDistancia de Mahalanobis: {distance:.4f}")


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

    # Ejemplo de PRN complejo
    complex_prn = ComplexPRN(0.6, 0.8, algorithm_type="quantum")

    print("\n=== Ejemplo de PRN ===")
    print(f"PRN original: {prn}")
    print(f"PRN secundario: {prn2}")
    print(f"PRN combinado: {combined_prn}")
    print(f"PRN complejo: {complex_prn}")

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


def main():
    """Función principal que ejecuta todas las demostraciones."""
    print("===== DEMOSTRACIÓN DE FUNCIONALIDADES =====")

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

    # Crear configuración personalizada para BayesLogic
    config = BayesLogicConfig(
        epsilon=1e-8,
        high_entropy_threshold=0.75,
        high_coherence_threshold=0.65,
        action_threshold=0.55
    )

    # Instanciar BayesLogic con la configuración personalizada
    bayes_logic = BayesLogic(config)

    # Calcular la acción a tomar
    result = bayes_logic.calculate_probabilities_and_select_action(
        entropy, coherence, prn.influence, previous_action
    )

    # Determinar la acción final
    action_to_take = result["action_to_take"]

    print(f"Acción a tomar: {'Move Right' if action_to_take == 1 else 'Move Left'}")
    print(f"Probabilidad condicional: {result['conditional_action_given_b']:.4f}")
    print(f"Umbral para decisión: {config.action_threshold}")


if __name__ == "__main__":
    main()