# =============================================================================
# 1. Importaciones y Configuración
# =============================================================================
import numpy as np
import torch  # Usado en inicializadores FFT
import matplotlib.pyplot as plt  # Usado para visualización
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from sklearn.covariance import EmpiricalCovariance  # Usado para Mahalanobis
from dataclasses import dataclass  # Usado para TimeSeries y NeuralNode (visualización)
import os
import sys
from pathlib import Path
from scipy import stats  # Para cálculos estadísticos avanzados
import warnings

# *** NOTA IMPORTANTE: Dependencias Externas ***
# Las clases QuantumBayesMahalanobis y FFTBayesIntegrator 
# pueden requerir bibliotecas como TensorFlow/TensorFlow Probability.
# Si planeas usarlas, asegúrate de instalarlas y descomenta las líneas:
# import tensorflow as tf
# import tensorflow_probability as tfp # Usado en tfp.stats.covariance

# Configuración unificada de logging (adaptada del segundo snippet)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Salida a consola
        # logging.FileHandler("cuadrante_coremind_unified.log") # Opcional: guardar log en archivo
    ]
)
logger = logging.getLogger(__name__)

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log2")

# =============================================================================
# 2. Funciones de Utilidad (Versiones robustas y unificadas)
# =============================================================================

def shannon_entropy(data: Union[List[Any], np.ndarray, Dict[Any, float]]) -> float:
    """
    Calcula la entropía de Shannon de un conjunto de datos.

    Puede manejar listas/arrays de valores o diccionarios de probabilidades.
    Se unifica la lógica de ambos snippets y se usa la versión del primer
    snippet (basada en probabilidades) como enfoque principal si se proporciona un dict.
    Si se proporciona una lista/array, calcula la distribución de probabilidad empírica.

    Args:
        data (Union[List[Any], np.ndarray, Dict[Any, float]]): Datos o diccionario de probabilidades.

    Returns:
        float: Entropía de Shannon en bits.

    Raises:
        ValueError: Si data está vacío o tiene formato inválido.
    """
    if not data:
        return 0.0  # Entropía de un conjunto vacío es 0 por convención aquí

    if isinstance(data, dict):
        # Asume que el diccionario ya contiene probabilidades válidas
        probs = np.array(list(data.values()), dtype=float)
        if not np.isclose(np.sum(probs), 1.0, rtol=1e-5) or np.any(probs < 0):
            logger.warning("Las 'probabilidades' en el diccionario no suman 1 o contienen negativos. Normalizando.")
            total_sum = np.sum(probs)
            if total_sum <= 0:
                return 0.0  # Evitar división por cero o log de negativos
            probs = probs / total_sum

    elif isinstance(data, (list, np.ndarray)):
        # Calcula la distribución de probabilidad empírica
        data_array = np.array(data, dtype=float)
        if data_array.size == 0:
            return 0.0

        values, counts = np.unique(data_array, return_counts=True)
        probs = counts / len(data_array)

    else:
        raise TypeError("El tipo de datos debe ser una lista, array de numpy o diccionario de probabilidades.")

    # Filtrar valores donde prob es 0 para evitar log(0)
    probs = probs[probs > 0]

    # Calcular entropía
    if probs.size == 0:
        return 0.0  # Si todos los valores eran 0 o filtrados

    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def calculate_cosines(metric1: float, metric2: float) -> Tuple[float, float, float]:
    """
    Calcula cosenos directores (x, y, z) para un vector tridimensional
    basado en dos métricas de entrada (ej. entropía y coherencia).
    
    Args:
        metric1 (float): Primera métrica (ej. entropía)
        metric2 (float): Segunda métrica (ej. coherencia)
        
    Returns:
        Tuple[float, float, float]: Cosenos directores (x, y, z)
    """
    # Asegurar valores no negativos para evitar problemas con sqrt
    val1 = abs(metric1)
    val2 = abs(metric2)

    # Añadir epsilon para evitar magnitud cero si ambas métricas son cero
    epsilon = 1e-10
    magnitude = np.sqrt(val1**2 + val2**2 + 1 + epsilon)  # El +1 le da una dimensión base no nula

    cos_x = val1 / magnitude
    cos_y = val2 / magnitude
    cos_z = 1.0 / magnitude  # El tercer componente se deriva para cerrar el vector 3D

    return cos_x, cos_y, cos_z


def calculate_wave_coherence(wave_data: np.ndarray) -> float:
    """
    Calcula la coherencia de una onda basada en su autocorrelación normalizada.
    
    Args:
        wave_data (np.ndarray): Los datos de la onda
        
    Returns:
        float: Valor de coherencia entre 0 y 1
    """
    if wave_data.size <= 1:
        return 0.0
    
    # Normalizar los datos de la onda
    normalized_wave = (wave_data - np.mean(wave_data)) / (np.std(wave_data) + 1e-10)
    
    # Calcular la autocorrelación (usando correlación de Pearson)
    n = len(normalized_wave)
    if n <= 1:
        return 0.0
        
    # Usar sólo los primeros N/2 lags para una mejor medida de coherencia
    max_lag = n // 2
    auto_corr = np.correlate(normalized_wave, normalized_wave, mode='full')[n-1:n+max_lag] / n
    
    # La coherencia es el promedio de la autocorrelación normalizada
    coherence = np.mean(np.abs(auto_corr)) if auto_corr.size > 0 else 0.0
    
    # Ajustar a rango [0,1]
    coherence = max(0.0, min(1.0, coherence))
    
    return coherence


def mahalanobis_distance(x: np.ndarray, data: np.ndarray) -> float:
    """
    Calcula la distancia de Mahalanobis entre un punto y un conjunto de datos.
    
    Args:
        x (np.ndarray): Vector para el cual calcular la distancia
        data (np.ndarray): Matriz de datos de referencia (cada fila es un punto)
        
    Returns:
        float: Distancia de Mahalanobis
    """
    if data.size == 0 or x.size == 0:
        return np.inf
    
    try:
        # Asegurar que x y data son arrays de numpy con dimensiones apropiadas
        x = np.atleast_1d(x).flatten()
        data = np.atleast_2d(data)
        
        if data.shape[1] != x.shape[0]:
            # Asegurar compatibilidad de dimensiones
            if data.shape[0] == x.shape[0] and data.shape[1] != x.shape[0]:
                # Transponer data si parece que las dimensiones están invertidas
                data = data.T
            elif data.shape[1] != x.shape[0]:
                logger.error(f"Dimensiones incompatibles: x={x.shape}, data={data.shape}")
                return np.inf
        
        # Calcular media y matriz de covarianza
        mean = np.mean(data, axis=0)
        cov = EmpiricalCovariance().fit(data)
        inv_cov = cov.precision_
        
        # Calcular distancia de Mahalanobis
        diff = x - mean
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
        
        return float(dist)
    except Exception as e:
        logger.error(f"Error en cálculo de Mahalanobis: {e}")
        return np.inf


# =============================================================================
# 3. Clases de Análisis del Ruido (El PRN Analizador)
# =============================================================================
class AnalysisPRN:
    """
    Clase principal para modelar y analizar el Ruido Probabilístico de Records.
    
    Esta clase se "alimenta" de los datos ruidosos generados por la simulación
    o un sistema real, los analiza usando Mahalanobis, Shannon, etc., y registra
    las métricas clave para la posterior secuencia de la RNN.
    """
    def __init__(self, analysis_influence: float = 0.5, history_window: int = 10, **parameters):
        """
        Inicializa el analizador PRN.

        Args:
            analysis_influence (float): Un factor que modula la sensibilidad del análisis.
            history_window (int): Tamaño de la ventana de historia para cálculos basados en tendencias.
            **parameters: Parámetros adicionales para configuraciones de análisis.
        """
        if not 0 <= analysis_influence <= 1:
            logger.warning(f"Valor de analysis_influence fuera de rango: {analysis_influence}, ajustando a [0, 1]")
            analysis_influence = max(0, min(1, analysis_influence))

        self.analysis_influence = analysis_influence
        self.history_window = max(1, history_window)
        self.parameters = parameters
        self.analysis_records_history = []  # Historial de métricas analizadas
        self.vector_history = []  # Historial de representaciones vectoriales para Mahalanobis
        
        # Coeficientes para cálculos de tendencia
        self.trend_coeffs = {
            'entropy': 0.0,
            'coherence': 0.0,
            'mahalanobis': 0.0
        }

    def record_noise_metrics(self,
                            simulated_process_data: Dict[str, Any],
                            reference_distribution_data: Optional[np.ndarray] = None
                           ) -> Dict[str, Any]:
        """
        Analiza los datos ruidosos de un paso de simulación/proceso y registra métricas.

        Args:
            simulated_process_data (Dict[str, Any]): Datos de salida del proceso simulado.
            reference_distribution_data (Optional[np.ndarray]): Datos de referencia para Mahalanobis.

        Returns:
            Dict[str, Any]: Un diccionario de métricas calculadas.
        """
        metrics = {}
        
        # --- 3.1. Calcular Entropía ---
        if 'simulated_probabilities' in simulated_process_data and simulated_process_data['simulated_probabilities']:
            metrics['shannon_entropy'] = shannon_entropy(simulated_process_data['simulated_probabilities'])
        elif 'raw_wave_representation' in simulated_process_data and simulated_process_data['raw_wave_representation'] is not None:
            # Calcular entropía de la distribución de valores en la onda
            metrics['wave_entropy'] = shannon_entropy(simulated_process_data['raw_wave_representation'])
        
        # --- 3.2. Calcular Coherencia de la onda ---
        if 'raw_wave_representation' in simulated_process_data and simulated_process_data['raw_wave_representation'] is not None:
            metrics['wave_coherence'] = calculate_wave_coherence(simulated_process_data['raw_wave_representation'])
        
        # --- 3.3. Calcular Mahalanobis ---
        if 'vector_representation' in simulated_process_data and simulated_process_data['vector_representation'] is not None:
            vector = simulated_process_data['vector_representation']
            
            # Guardar el vector en el historial para futuros cálculos
            self.vector_history.append(vector)
            
            # Limitar el tamaño del historial
            if len(self.vector_history) > self.history_window:
                self.vector_history = self.vector_history[-self.history_window:]
            
            # Si tenemos suficientes vectores en el historial o tenemos datos de referencia
            if len(self.vector_history) > 1 or (reference_distribution_data is not None and reference_distribution_data.size > 0):
                try:
                    # Usar datos de referencia si existen, o el historial propio
                    ref_data = reference_distribution_data if reference_distribution_data is not None else np.array(self.vector_history[:-1])
                    
                    # Calcular la distancia de Mahalanobis
                    mahal_dist = mahalanobis_distance(vector, ref_data)
                    metrics['mahalanobis_distance'] = mahal_dist
                except Exception as e:
                    logger.error(f"Error calculando Mahalanobis: {e}")
                    metrics['mahalanobis_distance'] = np.nan
        
        # --- 3.4. Incorporar otras métricas de interés ---
        for key in ['action_taken', 'collapsed_state_feature', 'external_noise_influence_at_step']:
            if key in simulated_process_data:
                # Simplificar nombres de claves
                simple_key = key.replace('_at_step', '')
                metrics[simple_key] = simulated_process_data[key]
        
        # Añadir número de iteración si está disponible
        if 'iteration' in simulated_process_data:
            metrics['iteration'] = simulated_process_data['iteration']
        
        # --- 3.5. Calcular tendencias si tenemos suficiente historial ---
        if len(self.analysis_records_history) >= 2:
            self._update_trend_metrics(metrics)
        
        # --- 3.6. Registrar la historia ---
        self.analysis_records_history.append(metrics)
        
        logger.debug(f"Métricas de ruido registradas: {metrics}")
        return metrics
    
    def _update_trend_metrics(self, current_metrics: Dict[str, Any]) -> None:
        """
        Actualiza las métricas de tendencia basadas en el historial.
        
        Args:
            current_metrics (Dict[str, Any]): Métricas actuales para actualizar con tendencias
        """
        # Límite de historia a considerar
        history_limit = min(self.history_window, len(self.analysis_records_history))
        
        if history_limit < 2:
            return  # No hay suficiente historia para calcular tendencia
        
        # Extraer valores de métricas relevantes del historial
        history = self.analysis_records_history[-history_limit:]
        
        # Calcular tendencias para métricas clave disponibles
        for key in ['shannon_entropy', 'wave_entropy', 'wave_coherence', 'mahalanobis_distance']:
            if key in current_metrics and all(key in record for record in history):
                values = [record[key] for record in history]
                
                if len(values) >= 2:
                    # Calcular coeficiente de tendencia lineal
                    x = np.arange(len(values))
                    trend = np.polyfit(x, values, 1)[0]  # Pendiente de la regresión lineal
                    
                    # Normalizar el coeficiente de tendencia a [-1, 1]
                    max_trend = max(abs(max(values) - min(values)), 1e-6)
                    normalized_trend = np.clip(trend / max_trend, -1, 1)
                    
                    # Guardar la tendencia
                    trend_key = key.replace('shannon_', '').replace('wave_', '').replace('_distance', '')
                    self.trend_coeffs[trend_key] = normalized_trend
                    
                    # Añadir tendencia a las métricas actuales
                    current_metrics[f"{key}_trend"] = normalized_trend
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """
        Retorna el historial completo de métricas analizadas.
        
        Returns:
            List[Dict[str, Any]]: Historial de análisis
        """
        return self.analysis_records_history
    
    def get_mahalanobis_stats(self) -> Dict[str, float]:
        """
        Retorna estadísticas sobre las distancias de Mahalanobis registradas.
        
        Returns:
            Dict[str, float]: Estadísticas sobre Mahalanobis
        """
        mahal_values = [record.get('mahalanobis_distance', np.nan) 
                        for record in self.analysis_records_history]
        mahal_values = [v for v in mahal_values if not np.isnan(v)]
        
        if not mahal_values:
            return {'count': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        return {
            'count': len(mahal_values),
            'mean': np.mean(mahal_values),
            'std': np.std(mahal_values),
            'min': np.min(mahal_values),
            'max': np.max(mahal_values),
            'trend': self.trend_coeffs.get('mahalanobis', 0.0)
        }


class ComplexPRN(AnalysisPRN):
    """
    Extensión para ruido complejo con capacidades adicionales para analizar 
    patrones y componentes más sofisticados en los datos.
    """
    def __init__(self, analysis_influence: float = 0.5, history_window: int = 10, 
                 complexity_factor: float = 0.5, **parameters):
        """
        Inicializa el analizador de PRN complejo.
        
        Args:
            analysis_influence (float): Factor que modula la sensibilidad
            history_window (int): Tamaño de la ventana de historia
            complexity_factor (float): Factor que modula la complejidad del análisis
            **parameters: Parámetros adicionales
        """
        super().__init__(analysis_influence, history_window, **parameters)
        self.complexity_factor = complexity_factor
        self.eigenvalues_history = []
    
    def record_noise_metrics(self, 
                             simulated_process_data: Dict[str, Any],
                             reference_distribution_data: Optional[np.ndarray] = None
                            ) -> Dict[str, Any]:
        """
        Versión extendida que analiza componentes complejos adicionales.
        
        Args:
            simulated_process_data: Datos del proceso simulado
            reference_distribution_data: Datos de referencia
            
        Returns:
            Dict[str, Any]: Métricas calculadas incluyendo componentes complejos
        """
        # Obtener métricas básicas usando la implementación de la clase padre
        metrics = super().record_noise_metrics(simulated_process_data, reference_distribution_data)
        
        # Añadir análisis de componentes complejos
        if 'vector_representation' in simulated_process_data and simulated_process_data['vector_representation'] is not None:
            vector = simulated_process_data['vector_representation']
            
            # Si el vector tiene suficiente dimensionalidad, aplicar PCA para extraer componentes
            if len(vector) > 3:
                try:
                    # Calcular autovalores y componentes principales usando SVD
                    if len(self.vector_history) > 1:
                        matrix = np.array(self.vector_history)
                        centered = matrix - np.mean(matrix, axis=0)
                        
                        # SVD para extraer componentes
                        U, s, Vh = np.linalg.svd(centered, full_matrices=False)
                        
                        # Guardar autovalores (valores singulares al cuadrado)
                        self.eigenvalues_history.append(s**2)
                        metrics['top_eigenvalue'] = float(s[0]**2)
                        
                        # Calcular la proporción de varianza explicada por el primer componente
                        explained_var_ratio = (s[0]**2) / sum(s**2)
                        metrics['dominant_component_ratio'] = float(explained_var_ratio)
                        
                        # Proyección en componentes principales
                        projection = np.dot(vector - np.mean(matrix, axis=0), Vh.T)
                        metrics['pc1_projection'] = float(projection[0])
                        metrics['pc2_projection'] = float(projection[1]) if len(projection) > 1 else 0.0
                except Exception as e:
                    logger.error(f"Error en análisis complejo de componentes: {e}")
        
        return metrics


# =============================================================================
# 4. Lógica Bayesiana
# =============================================================================
class BayesLogic:
    """
    Clase para lógica de inferencia Bayesiana.
    Puede ser usada tanto en el proceso de simulación (para la decisión de colapso)
    como en el análisis de ruido (para inferir sobre el estado o la naturaleza del ruido).
    """
    def __init__(self) -> None:
        self.EPSILON = 1e-6
        self.HIGH_ENTROPY_THRESHOLD = 0.8
        self.HIGH_COHERENCE_THRESHOLD = 0.6
        self.ACTION_THRESHOLD = 0.5
        
        # Factores adicionales para cálculos bayesianos
        self.coherence_weight = 1.0
        self.entropy_weight = 1.0
        self.noise_weight = 0.7
    
    def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
        """
        Calcula la probabilidad posterior P(A|B) = P(B|A)P(A)/P(B)
        
        Args:
            prior_a (float): Probabilidad a priori de A, P(A)
            prior_b (float): Probabilidad a priori de B, P(B)
            conditional_b_given_a (float): Probabilidad condicional P(B|A)
            
        Returns:
            float: Probabilidad posterior P(A|B)
        """
        prior_b = max(prior_b, self.EPSILON)
        return (conditional_b_given_a * prior_a) / prior_b

    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """
        Calcula la probabilidad condicional P(A|B) = P(A,B)/P(B)
        
        Args:
            joint_probability (float): Probabilidad conjunta P(A,B)
            prior (float): Probabilidad a priori P(B)
            
        Returns:
            float: Probabilidad condicional P(A|B)
        """
        prior = max(prior, self.EPSILON)
        return joint_probability / prior

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """
        Calcula la probabilidad a priori de entropía alta.
        
        Args:
            entropy (float): Valor de entropía
            
        Returns:
            float: Probabilidad de entropía alta
        """
        # Versión mejorada con transición suave en lugar de umbral abrupto
        return 0.1 + (0.3 - 0.1) * self._sigmoid(self.entropy_weight * (entropy - self.HIGH_ENTROPY_THRESHOLD))

    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """
        Calcula la probabilidad a priori de coherencia alta.
        
        Args:
            coherence (float): Valor de coherencia
            
        Returns:
            float: Probabilidad de coherencia alta
        """
        # Versión mejorada con transición suave en lugar de umbral abrupto
        return 0.2 + (0.6 - 0.2) * self._sigmoid(self.coherence_weight * (coherence - self.HIGH_COHERENCE_THRESHOLD))

    def calculate_joint_probability(self, metric1: float, metric2: float, influence: float) -> float:
        """
        Calcula la probabilidad conjunta basada en dos métricas y un factor de influencia.
        
        Args:
            metric1 (float): Primera métrica (ej. coherencia)
            metric2 (float): Segunda métrica (ej. otra métrica)
            influence (float): Factor de influencia [0,1]
            
        Returns:
            float: Probabilidad conjunta
        """
        # Versión mejorada con normalización y combinación ponderada
        metric1_norm = self._normalize_metric(metric1)
        metric2_norm = self._normalize_metric(metric2)
        
        # Combinación ponderada
        joint_prob = (metric1_norm * influence) + (metric2_norm * (1 - influence))
        
        # Factor de escala para asegurar que sea una probabilidad válida
        return max(0.0, min(1.0, joint_prob * 0.5))
    
    def _sigmoid(self, x: float) -> float:
        """
        Función sigmoide para transiciones suaves.
        
        Args:
            x (float): Valor de entrada
            
        Returns:
            float: Valor sigmoide entre 0 y 1
        """
        return 1.0 / (1.0 + np.exp(-x))
    
    def _normalize_metric(self, metric: float) -> float:
        """
        Normaliza una métrica para que esté en el rango [0,1].
        
        Args:
            metric (float): Métrica a normalizar
            
        Returns:
            float: Métrica normalizada
        """
        # Asumir que las métricas típicas están en [0,1] o cerca, pero ajustar si es necesario
        return max(0.0, min(1.0, metric))

    def calculate_probabilities_and_select_action(self, **metrics) -> Dict[str, Any]:
        """
        Calcula probabilidades y selecciona una "acción" usando lógica bayesiana.
        
        Args:
            **metrics: Diccionario de métricas (del análisis PRN u otras fuentes)
            
        Returns:
            Dict[str, Any]: Probabilidades calculadas y acción seleccionada
        """
        # Extraer métricas relevantes del diccionario, usando valores predeterminados si no existen
        entropy = metrics.get('shannon_entropy', metrics.get('wave_entropy', 0.1))
        noise_influence = metrics.get('external_noise_influence', 0.1)
        coherence = metrics.get('wave_coherence', 0.5)
        
        # Calcular priors bayesianos
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)
        
        # Conditional P(B|A): probabilidad de alta coherencia dado alta entropía
        # Modulada por noise_influence
        conditional_coherence_given_entropy = (
            self.noise_weight * noise_influence + (1 - self.noise_weight) * (1 - noise_influence)
            if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.2
        )
        
        # P(A|B): probabilidad posterior de alta entropía dada alta coherencia
        posterior_entropy_given_coherence = self.calculate_posterior_probability(
            high_entropy_prior, high_coherence_prior, conditional_coherence_given_entropy
        )
        
        # Cálculo mejorado de probabilidades de acción
        # Usar tendencias si están disponibles
        entropy_trend = metrics.get('shannon_entropy_trend', metrics.get('wave_entropy_trend', 0.0))
        coherence_trend = metrics.get('wave_coherence_trend', 0.0))
        
        # Base para prob_action_0 que incorpora todas las métricas
        base_prob_0 = 0.5 + (
            - 0.1 * entropy                    # A mayor entropía, menor prob_0
            + 0.1 * coherence                  # A mayor coherencia, mayor prob_0
            - 0.05 * noise_influence           # A mayor ruido externo, menor prob_0
            - 0.02 * entropy_trend             # Si entropía aumenta, menor prob_0
            + 0.02 * coherence_trend           # Si coherencia aumenta, mayor prob_0
        )
        
        # Asegurar que esté en rango [0,1]
        prob_action_0 = max(0.0, min(1.0, base_prob_0))
        prob_action_1 = 1.0 - prob_action_0
        
        # Normalizar por seguridad
        total = prob_action_0 + prob_action_1
        if total > 0:
            prob_action_0 /= total
            prob_action_1 /= total
        else:
            prob_action_0 = prob_action_1 = 0.5
        
        # Decisión probabilística
        action_to_take = 1 if np.random.random() < prob_action_1 else 0
        
        return {
            "calculated_probabilities": {
                "prob_action_0": prob_action_0,
                "prob_action_1": prob_action_1,
                "high_entropy_prior": high_entropy_prior,
                "high_coherence_prior": high_coherence_prior,
                "posterior_entropy_given_coherence": posterior_entropy_given_coherence
            },
            "selected_action": action_to_take,
            "confidence": max(prob_action_0, prob_action_1)
        }

    def update_weights(self, success_rate: float, learning_rate: float = 0.1) -> None:
        """
        Actualiza los pesos internos basándose en el éxito de las predicciones.
        
        Args:
            success_rate (float): Tasa de éxito de las predicciones [0,1]
            learning_rate (float): Tasa de aprendizaje para actualización de pesos
        """
        # Ajustar pesos basados en el éxito
        delta = learning_rate * (success_rate - 0.5)  # 0.5 como punto neutral
        
        self.coherence_weight = max(0.1, min(2.0, self.coherence_weight + delta))
        self.entropy_weight = max(0.1, min(2.0, self.entropy_weight + delta))
        self.noise_weight = max(0.1, min(1.0, self.noise_weight + delta))

    def get_state(self) -> Dict[str, float]:
        """
        Retorna el estado actual de los pesos y umbrales.
        
        Returns:
            Dict[str, float]: Estado actual de la lógica bayesiana
        """
        return {
            "coherence_weight": self.coherence_weight,
            "entropy_weight": self.entropy_weight,
            "noise_weight": self.noise_weight,
            "high_entropy_threshold": self.HIGH_ENTROPY_THRESHOLD,
            "high_coherence_threshold": self.HIGH_COHERENCE_THRESHOLD,
            "action_threshold": self.ACTION_THRESHOLD
        }

# =============================================================================
# 5. Integrador FFT Bayesiano
# =============================================================================
class FFTBayesIntegrator:
    """
    Integra análisis FFT con inferencia bayesiana para procesar señales.
    """
    def __init__(self, sampling_rate: float = 1.0, window_size: int = 256):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.bayes_logic = BayesLogic()
        self.fft_history = []
        
    def process_signal(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Procesa una señal usando FFT e inferencia bayesiana.
        
        Args:
            signal (np.ndarray): Señal de entrada
            
        Returns:
            Dict[str, Any]: Resultados del procesamiento
        """
        # Realizar FFT
        if len(signal) < self.window_size:
            # Padding con ceros si es necesario
            signal = np.pad(signal, (0, self.window_size - len(signal)))
            
        fft_result = np.fft.fft(signal, n=self.window_size)
        frequencies = np.fft.fftfreq(self.window_size, d=1/self.sampling_rate)
        
        # Calcular espectro de potencia
        power_spectrum = np.abs(fft_result)**2
        
        # Guardar en historial
        self.fft_history.append(power_spectrum)
        if len(self.fft_history) > 10:  # Mantener historial limitado
            self.fft_history.pop(0)
            
        # Calcular métricas espectrales
        dominant_freq = frequencies[np.argmax(power_spectrum)]
        spectral_entropy = shannon_entropy(power_spectrum)
        
        # Inferencia bayesiana sobre las características espectrales
        bayes_results = self.bayes_logic.calculate_probabilities_and_select_action(
            shannon_entropy=spectral_entropy,
            wave_coherence=self._calculate_spectral_coherence(),
            external_noise_influence=self._estimate_noise_influence(power_spectrum)
        )
        
        return {
            "fft_result": fft_result,
            "frequencies": frequencies,
            "power_spectrum": power_spectrum,
            "dominant_frequency": dominant_freq,
            "spectral_entropy": spectral_entropy,
            "bayes_analysis": bayes_results
        }
        
    def _calculate_spectral_coherence(self) -> float:
        """
        Calcula la coherencia espectral basada en el historial FFT.
        
        Returns:
            float: Valor de coherencia espectral
        """
        if len(self.fft_history) < 2:
            return 0.0
            
        # Calcular correlación entre espectros consecutivos
        correlations = []
        for i in range(len(self.fft_history)-1):
            corr = np.corrcoef(self.fft_history[i], self.fft_history[i+1])[0,1]
            correlations.append(corr)
            
        return np.mean(correlations) if correlations else 0.0
        
    def _estimate_noise_influence(self, power_spectrum: np.ndarray) -> float:
        """
        Estima la influencia del ruido en el espectro de potencia.
        
        Args:
            power_spectrum (np.ndarray): Espectro de potencia
            
        Returns:
            float: Estimación de la influencia del ruido [0,1]
        """
        # Calcular ratio señal/ruido simplificado
        signal_power = np.max(power_spectrum)
        noise_floor = np.median(power_spectrum)
        
        if signal_power <= noise_floor:
            return 1.0
            
        snr = 10 * np.log10(signal_power / noise_floor)
        
        # Convertir SNR a una medida de influencia de ruido entre 0 y 1
        noise_influence = 1.0 / (1.0 + np.exp(0.1 * (snr - 20)))  # 20dB como punto medio
        
        return noise_influence