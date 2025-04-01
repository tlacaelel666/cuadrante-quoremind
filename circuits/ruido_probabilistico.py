import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.covariance import EmpiricalCovariance

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PRN:
    """
    Clase para modelar el Ruido Probabilístico de Referencia (Probabilistic Reference Noise).
    
    Esta clase generalizada puede ser utilizada para representar cualquier tipo de
    influencia probabilística en un sistema.
    
    Atributos:
        influence (float): Factor de influencia entre 0 y 1.
        algorithm_type (str): Tipo de algoritmo a utilizar.
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
        self._validate_influence(influence)
        self.influence = influence
        self.algorithm_type = algorithm_type
        self.parameters = parameters
        self.history = []  # Historial de cambios en la influencia
    
    def _validate_influence(self, influence: float) -> None:
        """
        Valida que el valor de influencia esté dentro del rango permitido.
        
        Args:
            influence (float): Valor de influencia a validar.
            
        Raises:
            ValueError: Si influence está fuera del rango [0,1].
        """
        if not 0 <= influence <= 1:
            raise ValueError(f"La influencia debe estar entre 0 y 1. Valor recibido: {influence}")

    def adjust_influence(self, adjustment: float) -> float:
        """
        Ajusta el factor de influencia dentro de los límites permitidos.
        
        Args:
            adjustment (float): Valor de ajuste (positivo o negativo).
        
        Returns:
            float: El nuevo valor de influencia.
        """
        previous_influence = self.influence
        new_influence = self.influence + adjustment

        if not 0 <= new_influence <= 1:
            # Truncamos al rango válido
            new_influence = max(0, min(1, new_influence))
            logger.warning(f"Influencia ajustada a {new_influence} para mantenerla en el rango [0,1]")

        self.influence = new_influence
        
        # Registrar el cambio en el historial
        self.history.append({
            'previous': previous_influence,
            'adjustment': adjustment,
            'new': new_influence,
            'truncated': previous_influence + adjustment != new_influence
        })
        
        return new_influence

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
            TypeError: Si other_prn no es una instancia de PRN.
        """
        if not isinstance(other_prn, PRN):
            raise TypeError(f"El parámetro other_prn debe ser una instancia de PRN. Recibido: {type(other_prn)}")
            
        if not 0 <= weight <= 1:
            raise ValueError(f"El peso debe estar entre 0 y 1. Valor recibido: {weight}")

        # Combinación ponderada de las influencias
        combined_influence = self.influence * weight + other_prn.influence * (1 - weight)

        # Combinar los parámetros de ambos PRN (los parámetros del segundo sobrescriben en caso de conflicto)
        combined_params = {**self.parameters, **other_prn.parameters}

        # Elegir el tipo de algoritmo según el peso
        algorithm = self.algorithm_type if weight >= 0.5 else other_prn.algorithm_type

        return PRN(combined_influence, algorithm, **combined_params)
    
    def record_noise(self, probabilities: Dict[str, float]) -> float:
        """
        Calcula la entropía de Shannon a partir de un diccionario de probabilidades.
        
        Args:
            probabilities (Dict[str, float]): Diccionario con probabilidades.
            
        Returns:
            float: Entropía de Shannon calculada.
        """
        # Extraer solo los valores de probabilidad
        probs = np.array(list(probabilities.values()))
        
        # Filtrar valores mayores que cero para evitar log(0)
        probs = probs[probs > 0]
        
        # Calcular la entropía
        entropy = -np.sum(probs * np.log2(probs))
        
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
    
    def __repr__(self) -> str:
        """
        Representación para desarrolladores del objeto PRN.
        
        Returns:
            str: Representación detallada del objeto PRN.
        """
        return self.__str__()


class ComplexPRN(PRN):
    """
    Extensión de PRN que representa números complejos para modelar ruido con componentes real e imaginaria.
    
    Atributos:
        real_component (float): Componente real.
        imaginary_component (float): Componente imaginaria.
        influence (float): Módulo del número complejo.
    """
    def __init__(self, real_component: float, imaginary_component: float, algorithm_type: str = None, **parameters):
        """
        Inicializa un objeto ComplexPRN con componentes real e imaginaria.
        
        Args:
            real_component (float): Componente real.
            imaginary_component (float): Componente imaginaria.
            algorithm_type (str, opcional): Tipo de algoritmo a utilizar.
            **parameters: Parámetros adicionales específicos del algoritmo.
        """
        self.real_component = real_component
        self.imaginary_component = imaginary_component
        # Calculamos el módulo como la influencia
        influence = np.sqrt(real_component**2 + imaginary_component**2)
        # Normalizamos para que esté entre 0 y 1
        normalized_influence = min(1.0, influence)
        
        super().__init__(normalized_influence, algorithm_type, **parameters)
        
        # Guardamos los componentes originales
        self.parameters['real_component'] = real_component
        self.parameters['imaginary_component'] = imaginary_component
    
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
        return f"ComplexPRN(real={self.real_component:.4f}, imag={self.imaginary_component:.4f}, influence={self.influence:.4f}, phase={self.get_phase():.4f})"


class EnhancedPRN(PRN):
    """
    Extiende la clase PRN para registrar distancias de Mahalanobis y con ello
    definir un 'ruido cuántico' adicional en el sistema.
    
    Atributos:
        mahalanobis_records (list): Lista de registros de distancias de Mahalanobis.
    """
    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        """
        Constructor que permite definir la influencia y el tipo de algoritmo,
        además de inicializar una lista para conservar registros promedio de
        distancias de Mahalanobis.
        
        Args:
            influence (float): Factor de influencia entre 0 y 1 (por defecto 0.5).
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
            probabilities (Dict[str, float]): Diccionario de probabilidades (ej. {"0": 0.5, "1": 0.5, ...}).
            quantum_states (np.ndarray): Estados cuánticos (n_muestras, n_dimensiones).

        Returns:
            Tuple[float, float]: 
                - Entropía calculada a partir de probabilities.
                - Distancia promedio de Mahalanobis.
                
        Raises:
            ValueError: Si quantum_states está vacío o tiene forma incorrecta.
        """
        # Validación de entrada
        if quantum_states.size == 0:
            raise ValueError("El array de estados cuánticos no puede estar vacío")
        
        # Calculamos la entropía usando el método de la clase base
        entropy = self.record_noise(probabilities)

        try:
            # Ajuste del estimador de covarianza con manejo de errores
            if quantum_states.ndim < 2 or quantum_states.shape[0] < 2:
                raise ValueError(f"Se requieren al menos 2 muestras para estimar la covarianza. Forma actual: {quantum_states.shape}")
                
            cov_estimator = EmpiricalCovariance().fit(quantum_states)
            mean_state = np.mean(quantum_states, axis=0)
            
            # Usamos pseudoinversa para mayor estabilidad
            inv_cov = np.linalg.pinv(cov_estimator.covariance_)

            # Cálculo vectorizado de la distancia
            diff = quantum_states - mean_state
            aux = diff @ inv_cov
            dist_sqr = np.einsum('ij,ij->i', aux, diff)
            distances = np.sqrt(dist_sqr)
            mahal_mean = np.mean(distances)
            
            # Detectar valores atípicos
            std_dist = np.std(distances)
            outliers = distances > (mahal_mean + 3 * std_dist)
            
            # Registrar información adicional
            record_info = {
                'mean': mahal_mean,
                'std': std_dist,
                'min': np.min(distances),
                'max': np.max(distances),
                'outliers_count': np.sum(outliers),
                'entropy': entropy,
                'sample_count': len(distances)
            }

            # Se registra la información completa
            self.mahalanobis_records.append(record_info)
            
            return entropy, mahal_mean
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Error en el cálculo de la distancia de Mahalanobis: {e}")
            # Retornar valores por defecto en caso de error
            self.mahalanobis_records.append({
                'error': str(e),
                'entropy': entropy
            })
            return entropy, 0.0
            
    def get_mahalanobis_stats(self) -> Dict[str, float]:
        """
        Calcula estadísticas sobre los registros de distancias de Mahalanobis.
        
        Returns:
            Dict[str, float]: Diccionario con estadísticas (media, desviación estándar, etc.)
        """
        if not self.mahalanobis_records:
            return {"count": 0}
            
        # Filtramos solo los registros que tienen la clave 'mean'
        valid_records = [r for r in self.mahalanobis_records if isinstance(r, dict) and 'mean' in r]
        
        if not valid_records:
            return {"count": 0, "valid_count": 0}
            
        means = [r['mean'] for r in valid_records]
        
        return {
            "count": len(self.mahalanobis_records),
            "valid_count": len(valid_records),
            "global_mean": np.mean(means),
            "global_std": np.std(means),
            "min": np.min(means),
            "max": np.max(means),
            "last": means[-1] if means else None
        }


def shannon_entropy(data: List[Union[int, float]]) -> float:
    """
    Calcula la entropía de Shannon de un conjunto de datos.

    Args:
        data (List[Union[int, float]]): Lista de datos.

    Returns:
        float: Entropía de Shannon en bits.
        
    Raises:
        ValueError: Si data está vacío.
    """
    if not data:
        raise ValueError("El conjunto de datos no puede estar vacío")
        
    # 1. Convertir a numpy array si es necesario
    data_array = np.array(data)
    
    # 2. Contar ocurrencias de cada valor único:
    values, counts = np.unique(data_array, return_counts=True)

    # 3. Calcular probabilidades:
    probabilities = counts / len(data_array)

    # 4. Evitar logaritmos de cero:
    probabilities = probabilities[probabilities > 0]

    # 5. Calcular la entropía:
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def calculate_cosines(entropy: float, env_value: float) -> Tuple[float, float, float]:
    """
    Calcula los cosenos directores (x, y, z) para un vector tridimensional.

    Args:
        entropy (float): Entropía de Shannon (x).
        env_value (float): Valor contextual del entorno (y).

    Returns:
        Tuple[float, float, float]: Cosenos directores (cos_x, cos_y, cos_z).
    """
    # Asegurar valores positivos para evitar división por cero:
    epsilon = 1e-10  # Valor pequeño para evitar división por cero
    entropy = max(entropy, epsilon)
    env_value = max(env_value, epsilon)

    # Magnitud del vector tridimensional:
    magnitude = np.sqrt(entropy ** 2 + env_value ** 2 + 1)

    # Cálculo de cosenos directores:
    cos_x = entropy / magnitude
    cos_y = env_value / magnitude
    cos_z = 1 / magnitude

    return cos_x, cos_y, cos_z


class BayesLogic:
    """
    Clase para implementar lógica bayesiana utilizada en los cálculos de FFTBayesIntegrator.
    (Implementación simplificada)
    """
    def __init__(self):
        pass

    # Métodos adicionales de lógica bayesiana se implementarían aquí


class StatisticalAnalysis:
    """
    Clase para realizar análisis estadísticos utilizados en los cálculos de FFTBayesIntegrator.
    """
    def __init__(self):
        pass
        
    def shannon_entropy(self, data: List[Union[int, float]]) -> float:
        """
        Calcula la entropía de Shannon de un conjunto de datos.
        
        Args:
            data (List[Union[int, float]]): Lista de datos.
            
        Returns:
            float: Entropía de Shannon en bits.
        """
        return shannon_entropy(data)


class FFTBayesIntegrator:
    """
    Clase que integra la Transformada Rápida de Fourier (FFT) con el análisis bayesiano
    para procesar señales cuánticas y generar representaciones para la inicialización informada
    de modelos o como features para redes neuronales.
    """
    def __init__(self) -> None:
        """
        Inicializa el integrador FFT-Bayes con las clases necesarias y caché.
        """
        # Inicializa instancias de lógica bayesiana y análisis estadístico, además de una caché.
        self.bayes_logic = BayesLogic()
        self.stat_analysis = StatisticalAnalysis()
        self.cache: Dict[int, Dict[str, Union[np.ndarray, float]]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def clear_cache(self) -> None:
        """
        Limpia la caché de resultados calculados.
        """
        self.cache.clear()
        logger.info("Caché limpiada")

    def process_quantum_circuit(self, quantum_circuit: "ResilientQuantumCircuit") -> Dict[str, Union[np.ndarray, float]]:
        """
        Procesa un circuito cuántico resistente aplicando la FFT a su estado.
        
        Args:
            quantum_circuit (ResilientQuantumCircuit): Circuito cuántico a procesar.
            
        Returns:
            Dict[str, Union[np.ndarray, float]]: Resultados del procesamiento FFT.
        """
        try:
            # Obtiene las amplitudes complejas del estado.
            amplitudes = quantum_circuit.get_complex_amplitudes()
            # Procesa las amplitudes usando FFT.
            return self.process_quantum_state(amplitudes)
        except Exception as e:
            logger.exception(f"Error al procesar el circuito cuántico: {e}")
            raise

    def process_quantum_state(self, quantum_state: List[complex]) -> Dict[str, Union[np.ndarray, float]]:
        """
        Procesa un estado cuántico aplicando la FFT y extrayendo características frecuenciales.
        
        Args:
            quantum_state (List[complex]): Estado cuántico representado como lista de números complejos.
            
        Returns:
            Dict[str, Union[np.ndarray, float]]: Resultados del procesamiento FFT.
            
        Raises:
            ValueError: Si el estado cuántico está vacío.
            TypeError: Si el estado cuántico no es válido.
        """
        if not quantum_state:
            msg = "El estado cuántico no puede estar vacío."
            logger.error(msg)
            raise ValueError(msg)

        # Usa caché para evitar cálculos repetidos si el estado ya fue procesado.
        state_hash = hash(tuple(quantum_state))
        if state_hash in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit! Reutilizando resultado para hash {state_hash}")
            return self.cache[state_hash]
            
        self.cache_misses += 1

        try:
            # Convierte la lista en un array de complejos.
            quantum_state_array = np.array(quantum_state, dtype=complex)
        except Exception as e:
            logger.exception("Error al convertir el estado cuántico a np.array")
            raise TypeError(f"Estado cuántico inválido: {e}") from e

        # Aplica la FFT al estado cuántico.
        fft_result = np.fft.fft(quantum_state_array)
        # Calcula las magnitudes de la FFT.
        fft_magnitudes = np.abs(fft_result)
        # Calcula las fases de la FFT.
        fft_phases = np.angle(fft_result)
        
        # Calcula estadísticas adicionales
        # Calcula la entropía de Shannon.
        entropy = self.stat_analysis.shannon_entropy(fft_magnitudes.tolist())
        # Calcula la varianza de las fases.
        phase_variance = np.var(fft_phases)
        # Deriva una medida de coherencia a partir de la varianza.
        coherence = np.exp(-phase_variance)
        # Calcula la energía de la señal
        energy = np.sum(np.abs(quantum_state_array)**2)
        # Calcula la frecuencia dominante
        dominant_freq_idx = np.argmax(fft_magnitudes[1:]) + 1 if len(fft_magnitudes) > 1 else 0
        dominant_freq = dominant_freq_idx / len(quantum_state)

        result = {
            'magnitudes': fft_magnitudes,
            'phases': fft_phases,
            'entropy': entropy,
            'coherence': coherence,
            'energy': energy,
            'dominant_freq': dominant_freq,
            'dominant_freq_idx': dominant_freq_idx,
            'timestamp': np.datetime64('now')
        }
        
        # Almacena el resultado en la caché.
        self.cache[state_hash] = result
        return result

    def fft_based_initializer(self, quantum_state: List[complex], out_dimension: int, scale: float = 0.01) -> torch.Tensor:
        """
        Inicializa una matriz de pesos basada en la FFT del estado cuántico.
        
        Args:
            quantum_state (List[complex]): Estado cuántico a procesar.
            out_dimension (int): Dimensión de salida deseada para la matriz.
            scale (float, opcional): Factor de escala para los pesos. Por defecto 0.01.
            
        Returns:
            torch.Tensor: Matriz de pesos inicializada.
        """
        # Convierte el estado cuántico en un array.
        signal = np.array(quantum_state)
        # Aplica la FFT.
        fft_result = np.fft.fft(signal)
        # Obtiene las magnitudes.
        magnitudes = np.abs(fft_result)
        # Normaliza las magnitudes.
        norm_magnitudes = magnitudes / np.sum(magnitudes) if np.sum(magnitudes) > 0 else magnitudes
        # Crea una matriz replicando el vector.
        weight_matrix = scale * np.tile(norm_magnitudes, (out_dimension, 1))
        # Convierte la matriz a tensor de PyTorch.
        return torch.tensor(weight_matrix, dtype=torch.float32)

    def advanced_fft_initializer(
        self, 
        quantum_state: List[complex], 
        out_dimension: int, 
        in_dimension: Optional[int] = None,
        scale: float = 0.01, 
        use_phases: bool = True
    ) -> torch.Tensor:
        """
        Inicializador avanzado que crea una matriz rectangular utilizando magnitudes y fases de la FFT.
        
        Args:
            quantum_state (List[complex]): Estado cuántico a procesar.
            out_dimension (int): Dimensión de salida de la matriz.
            in_dimension (Optional[int], opcional): Dimensión de entrada de la matriz. 
                                                  Si es None, se usa la longitud del estado cuántico.
            scale (float, opcional): Factor de escala para los pesos. Por defecto 0.01.
            use_phases (bool, opcional): Si es True, incorpora información de fase. Por defecto True.
            
        Returns:
            torch.Tensor: Matriz de pesos inicializada.
        """
        # Validación de entrada
        if not quantum_state:
            raise ValueError("El estado cuántico no puede estar vacío")
            
        # Convierte el estado cuántico en un array.
        signal = np.array(quantum_state)
        # Define la dimensión de entrada si no se especifica.
        in_dimension = in_dimension or len(quantum_state)
        # Aplica la FFT.
        fft_result = np.fft.fft(signal)
        # Obtiene las magnitudes y fases.
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Normaliza las magnitudes para evitar división por cero.
        sum_magnitudes = np.sum(magnitudes)
        norm_magnitudes = magnitudes / sum_magnitudes if sum_magnitudes > 0 else np.ones_like(magnitudes) / len(magnitudes)

        # Construye el vector base para la matriz tomando la cantidad adecuada de características.
        if len(quantum_state) >= in_dimension:
            base_features = norm_magnitudes[:in_dimension]
            phase_features = phases[:in_dimension] if use_phases else None
        else:
            repeats = int(np.ceil(in_dimension / len(quantum_state)))
            base_features = np.tile(norm_magnitudes, repeats)[:in_dimension]
            phase_features = np.tile(phases, repeats)[:in_dimension] if use_phases else None

        if use_phases and phase_features is not None:
            # Incorpora la información de fase en las características con un factor de modulación.
            phase_modulation = 1 + 0.1 * np.cos(phase_features)
            base_features = base_features * phase_modulation

        # Crea la matriz de pesos desplazando el vector base para cada fila.
        weight_matrix = np.empty((out_dimension, in_dimension))
        for i in range(out_dimension):
            shift = i % len(base_features)
            weight_matrix[i] = np.roll(base_features, shift)
            
        # Normalización para evitar valores extremos
        max_abs = np.max(np.abs(weight_matrix))
        if max_abs > 0:
            weight_matrix = scale * weight_matrix / max_abs
        else:
            # Si todos los valores son cero, inicializar con valores pequeños aleatorios
            weight_matrix = scale * np.random.randn(out_dimension, in_dimension) / 100
            
        # Convierte la matriz a tensor de PyTorch.
        return torch.tensor(weight_matrix, dtype=torch.float32)


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de prueba
    sample_data = [1, 2, 3, 4, 5, 5, 2]
    
    # Calcula la entropía de Shannon
    entropy_value = shannon_entropy(sample_data)
    env_value = 0.8  # Ejemplo de un valor de entorno

    # Calcula los cosenos directores
    cos_x, cos_y, cos_z = calculate_cosines(entropy_value, env_value)

    # Muestra los resultados
    print(f"Entropía: {entropy_value:.4f}")
    print(f"Cosenos directores: cos_x = {cos_x:.4f}, cos_y = {cos_y:.4f}, cos_z = {cos_z:.4f}")
    
    # Creación y uso de PRN
    basic_prn = PRN(0.7, "gaussian", sigma=1.5, mean=0)
    print(f"PRN básico: {basic_prn}")
    
    # Demostración de ComplexPRN
    complex_prn = ComplexPRN(0.6, 0.8, "quantum", dimension=5)
    print(f"PRN complejo: {complex_prn}")
    
    # Demostración de EnhancedPRN con ruido cuántico
    enhanced_prn = EnhancedPRN(0.5, "mahalanobis", dimension=3)
    
    # Simular estados cuánticos y probabilidades
    n_samples = 20
    n_dims = 4
    quantum_states = np.random.random((n_samples, n_dims))
    probabilities = {"0": 0.3, "1": 0.7}
    
    # Registrar ruido cuántico
    try:
        entropy, mahal_dist = enhanced_prn.record_quantum_noise(probabilities, quantum_states)
        print(f"Entropía calculada: {entropy:.4f}, Distancia de Mahalanobis: {mahal_dist:.4f}")
        
        # Mostrar estadísticas
        stats = enhanced_prn.get_mahalanobis_stats()
        print("Estadísticas de Mahalanobis:", stats)
    except Exception as e:
        print(f"Error al registrar ruido cuántico: {e}")