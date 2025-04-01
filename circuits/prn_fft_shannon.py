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

         # Clase PRN modificada para representar números complejos.
    """
    def __init__(self, real_component: float, imaginary_component: float, algorithm_type: str = None, **parameters):
        self.real_component = real_component
        self.imaginary_component = imaginary_component
        self.influence = np.sqrt(real_component**2 + imaginary_component**2) #Modulo del numero complejo.
        self.algorithm_type = algorithm_type
        self.parameters = parameters
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

class EnhancedPRN(PRN):
    """
    Extiende la clase PRN para registrar distancias de Mahalanobis y con ello
    definir un 'ruido cuántico' adicional en el sistema.
    """
    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        """
        Constructor que permite definir la influencia y el tipo de algoritmo,
        además de inicializar una lista para conservar registros promedio de
        distancias de Mahalanobis.
        """
        super().__init__(influence, algorithm_type, **parameters)
        self.mahalanobis_records = []

def shannon_entropy(data: list) -> float:
    """
    Calculates the Shannon entropy of a data set.

    Args:
      data (list or numpy.ndarray): List or array of data.

    Returns:
      float: Shannon entropy in bits.
    """
    # 1. Count occurrences of each unique value:
    values, counts = np.unique(data, return_counts=True)

    # 2. Calculate probabilities:
    probabilities = counts / len(data)

    # 3. Avoid logarithms of zero:
    probabilities = probabilities[probabilities > 0]

    # 4. Calculate the entropy:
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def calculate_cosines(entropy: float, env_value: float) -> Tuple[float, float, float]:
    """
    Calculates the directional cosines (x, y, z) for a three-dimensional vector.

    Args:
      entropy (float): Shannon entropy (x).
      env_value (float): Contextual environment value (y).

    Returns:
      tuple: Directional cosines (cos_x, cos_y, cos_z).
    """
    # Ensure to avoid division by zero:
    if entropy == 0:
        entropy = 1e-6
    if env_value == 0:
        env_value = 1e-6

    # Magnitude of the three-dimensional vector:
    magnitude = np.sqrt(entropy ** 2 + env_value ** 2 + 1)

    # Calculation of directional cosines:
    cos_x = entropy / magnitude
    cos_y = env_value / magnitude
    cos_z = 1 / magnitude

    return cos_x, cos_y, cos_z

# Example of usage:
if __name__ == "__main__":
    # Test data:
    sample_data = [1, 2, 3, 4, 5, 5, 2]
    entropy = shannon_entropy(sample_data)
    env_value = 0.8  # Example of an environment value

    cos_x, cos_y, cos_z = calculate_cosines(entropy, env_value)

    print(f"Entropy: {entropy:.4f}")
    print(f"Directional cosines: cos_x = {cos_x:.4f}, cos_y = {cos_y:.4f}, cos_z = {cos_z:.4f}")

    def record_quantum_noise(self, probabilities: dict, quantum_states: np.ndarray):
        """
        Registra un 'ruido cuántico' basado en la distancia de Mahalanobis
        calculada para los estados cuánticos.

        Parámetros:
        -----------
        probabilities: dict
            Diccionario de probabilidades (ej. {"0": p_0, "1": p_1, ...}).
        quantum_states: np.ndarray
            Estados cuánticos (n_muestras, n_dimensiones).

        Retorna:
        --------
        (entropy, mahal_mean): Tuple[float, float]
            - Entropía calculada a partir de probabilities.
            - Distancia promedio de Mahalanobis.
        """
        # Calculamos la entropía (este método se asume en la clase base PRN o BayesLogic).
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


class FFTBayesIntegrator:
    """
    Clase que integra la Transformada Rápida de Fourier (FFT) con el análisis bayesiano
    para procesar señales cuánticas y generar representaciones para la inicialización informada
    de modelos o como features para redes neuronales.
    """
    def __init__(self) -> None:
        # Inicializa instancias de lógica bayesiana y análisis estadístico, además de una caché.
        self.bayes_logic = BayesLogic()
        self.stat_analysis = StatisticalAnalysis()
        self.cache: Dict[int, Dict[str, Union[np.ndarray, float]]] = {}  # Caché para almacenar resultados ya calculados.

    def process_quantum_circuit(self, quantum_circuit: "ResilientQuantumCircuit") -> Dict[str, Union[np.ndarray, float]]:
        # Procesa un circuito cuántico resistente aplicando la FFT a su estado.
        amplitudes = quantum_circuit.get_complex_amplitudes()  # Obtiene las amplitudes complejas del estado.
        return self.process_quantum_state(amplitudes)  # Procesa las amplitudes usando FFT.

    def process_quantum_state(self, quantum_state: List[complex]) -> Dict[str, Union[np.ndarray, float]]:
        # Procesa un estado cuántico aplicando la FFT y extrayendo características frecuenciales.
        if not quantum_state:
            msg = "El estado cuántico no puede estar vacío."
            logger.error(msg)
            raise ValueError(msg)

        # Usa caché para evitar cálculos repetidos si el estado ya fue procesado.
        state_hash = hash(tuple(quantum_state))
        if state_hash in self.cache:
            return self.cache[state_hash]

        try:
            quantum_state_array = np.array(quantum_state, dtype=complex)  # Convierte la lista en un array de complejos.
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
        # Inicializa una matriz de pesos basada en la FFT del estado cuántico.
        signal = np.array(quantum_state)  # Convierte el estado cuántico en un array.
        fft_result = np.fft.fft(signal)  # Aplica la FFT.
        magnitudes = np.abs(fft_result)  # Obtiene las magnitudes.
        norm_magnitudes = magnitudes / np.sum(magnitudes)  # Normaliza las magnitudes.
        weight_matrix = scale * np.tile(norm_magnitudes, (out_dimension, 1))  # Crea una matriz replicando el vector.
        return torch.tensor(weight_matrix, dtype=torch.float32)  # Convierte la matriz a tensor de PyTorch.

    def advanced_fft_initializer(self, quantum_state: List[complex], out_dimension: int, in_dimension: Optional[int] = None,
                                 scale: float = 0.01, use_phases: bool = True) -> torch.Tensor:
        # Inicializador avanzado que crea una matriz rectangular utilizando magnitudes y fases de la FFT.
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
        weight_matrix = scale * weight_matrix / np.max(np.abs(weight_matrix))  # Escala la matriz para normalizarla.
        return torch.tensor(weight_matrix, dtype=torch.float32)


