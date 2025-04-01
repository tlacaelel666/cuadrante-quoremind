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
