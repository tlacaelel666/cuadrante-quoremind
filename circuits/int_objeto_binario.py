# integration_with_interface.py

import numpy as np
from typing import Dict, Any
import logging

# Importa las clases de tus módulos
from fourier_transformed import (ResilientQuantumCircuit, FFTBayesIntegrator,
                                StatisticalAnalysis, BayesLogic)  # Asegúrate de que estén en el PATH
from rnn_coord import QuantumLearningAgent, QuantumState, RNNCoordinator  # Asegúrate de que estén en el PATH

# Configuración de logging (puedes usar la misma configuración que antes o adaptarla)
logger = logging.getLogger(__name__)

class AgentInterfaceIntegrator:
    """
    Clase para integrar el QuantumLearningAgent con la interfaz de usuario y
    el backend de procesamiento cuántico (ResilientQuantumCircuit y FFTBayesIntegrator).
    """

    def __init__(self) -> None:
        """
        Inicializa el integrador, creando instancias del agente, el circuito cuántico
        y el procesador FFT.
        """
        self.quantum_agent: QuantumLearningAgent = None  # Inicializado en setup_agent
        self.resilient_circuit: ResilientQuantumCircuit = None  # Inicializado en setup_quantum_circuit
        self.fft_integrator = FFTBayesIntegrator()
        self.bayes_logic = BayesLogic()
        self.stat_analysis = StatisticalAnalysis()


    def setup_agent(self, agent_settings: Dict[str, Any]) -> None:
        """
        Configura el QuantumLearningAgent con los parámetros proporcionados.

        Args:
            agent_settings (Dict[str, Any]):  Diccionario con la configuración del agente.
                Debe contener las claves: 'name', 'num_qubits', 'learning_rate',
                'input_size', 'hidden_size', 'rnn_type', 'dropout_rate', 'use_batch_norm',
                'adaptative_lr_decay'.
        """
        try:

            self.quantum_agent = QuantumLearningAgent(
                name=agent_settings['name'],
                num_qubits=agent_settings['num_qubits'],
                learning_rate=agent_settings['learning_rate'],
                input_size=agent_settings['input_size'],
                hidden_size=agent_settings['hidden_size'],
                rnn_type=agent_settings['rnn_type'],
                dropout_rate=agent_settings['dropout_rate'],
                use_batch_norm=agent_settings['use_batch_norm'],
                adaptative_lr_decay = agent_settings['adaptative_lr_decay']
            )
            logger.info("QuantumLearningAgent configurado correctamente.")
        except Exception as e:
            logger.exception("Error al configurar QuantumLearningAgent: %s", e)
            raise


    def setup_quantum_circuit(self, num_qubits: int) -> None:
        """
        Inicializa el circuito cuántico resistente.

        Args:
            num_qubits (int):  Número de qubits para el circuito.
        """
        try:
            self.resilient_circuit = ResilientQuantumCircuit(num_qubits=num_qubits)
            self.resilient_circuit.create_resilient_state()  # Crea el estado inicial
            logger.info("Circuito cuántico resistente inicializado con %d qubits.", num_qubits)
        except Exception as e:
            logger.exception("Error al inicializar el circuito cuántico: %s", e)
            raise

    def run_simulation(self, data: np.ndarray, sequence_length: int, num_iterations: int,
                       training_epochs:int, batch_size: int) -> None:
        """
        Ejecuta la simulación completa, integrando el agente, el circuito cuántico
        y el procesamiento FFT.

        Args:
            data (np.ndarray):  Datos de entrada para la simulación.
            sequence_length (int):  Longitud de secuencia para el entrenamiento de la RNN.
            num_iterations (int):  Número de iteraciones (episodios) de la simulación.
            training_epochs (int):  Número de épocas para el entrenamiento de la RNN.
            batch_size (int):  Tamaño del lote para el entrenamiento de la RNN.
        """

        if self.quantum_agent is None or self.resilient_circuit is None:
            logger.error("El agente o el circuito cuántico no han sido inicializados.")
            raise ValueError("El agente y el circuito cuántico deben ser inicializados antes de la simulación.")
        
        # Obtener el estado cuántico actual del circuito.
        current_quantum_state = self.resilient_circuit.get_complex_amplitudes()

        #Procesar estado cuantico con la FFT
        fft_results = self.fft_integrator.process_quantum_state(current_quantum_state)

        # Ejemplo de cómo usar los resultados de la FFT (puedes adaptarlo)
        magnitudes = fft_results['magnitudes']
        logger.info("Magnitudes FFT: %s", magnitudes)

        try:
            self.quantum_agent.simulate_quantum_rnn_interaction(
                data, sequence_length, num_iterations, training_epochs, batch_size
            )
            logger.info("Simulación completada con éxito.")
        except Exception as e:
            logger.exception("Error durante la simulación: %s", e)
            raise

    def apply_custom_gates(self, gates_list):
        """
        Aplica una lista personalizada de compuertas al circuito cuántico

        Args:
            gates_list: Lista de tuplas. Cada tupla: (nombre_compuerta, parametros, lista_qubits)
        """
        if self.resilient_circuit is None:
            logger.error("El circuito cuántico no ha sido inicializado.")
            raise ValueError("El circuito cuántico debe ser inicializado antes de aplicar compuertas.")

        try:
            self.resilient_circuit.apply_custom_gates(gates_list)
            logger.info("Compuertas personalizadas aplicadas con éxito")
        except Exception as e:
            logger.exception("Error al aplicar las compuertas personalizadas")
            raise

    def measure_qubit(self, qubit_index: int) -> None:
        """
        Realiza la medición de un qubit específico, preservando la coherencia de los demás.
        """
        if self.resilient_circuit is None:
            raise ValueError("El circuito cuántico debe ser inicializado primero")
        
        self.resilient_circuit.measure_qubit(qubit_index)
        logger.info(f"Qubit {qubit_index} medido, resto del estado preservado.")

    def measure_all_qubits(self) -> None:
        """
        Mide todos los qubits del circuito cuántico
        """

        if self.resilient_circuit is None:
            raise ValueError("El circuito cuántico debe ser inicializado primero")
        
        self.resilient_circuit.measure_all()
        logger.info("Todos los qubits medidos.")

    def get_quantum_state(self) -> Any:
        """
        Obtiene el estado cuántico actual del circuito
        """
        if self.resilient_circuit is None:
            raise ValueError("El circuito cuántico debe ser inicializado primero")
        
        return self.resilient_circuit.get_complex_amplitudes()

    def get_probabilities(self) -> Any:
        """
        Obtiene las probabilidades del estado cuántico
        """
        if self.resilient_circuit is None:
            raise ValueError("El circuito cuántico debe ser inicializado primero")
        return self.resilient_circuit.get_probabilities()

    def get_agent_settings(self) -> Dict:
        """
        Obtiene la configuración actual del agente
        """
        if self.quantum_agent is None:
            raise ValueError("El agente debe ser inicializado primero")

        return self.quantum_agent.get_settings()
    
    def update_agent_settings(self, new_settings: Dict) -> None:
        """
        Actualiza la configuración del agente
        """
        if self.quantum_agent is None:
            raise ValueError("El agente debe ser inicializado primero")
        self.quantum_agent.update_agent(new_settings)

    def calculate_bayes_metrics(self, entropy: float, coherence: float, prn_influence: float,
                                  action: int) -> Dict[str, float]:
        """Calcula métricas bayesianas basadas en las entradas proporcionadas.

        Args:
            entropy (float): Entropía del estado cuántico/sistema.
            coherence (float): Coherencia del estado cuántico/sistema.
            prn_influence (float): Influencia de un generador de números pseudoaleatorios o señal externa.
            action (int): Acción a evaluar (puede ser 0 o 1, por ejemplo).

        Returns:
            Dict[str, float]: Diccionario con las métricas bayesianas calculadas:
                - "action_to_take": Acción recomendada (0 o 1) según el umbral.
                - "high_entropy_prior": Probabilidad previa basada en la entropía.
                - "high_coherence_prior": Probabilidad previa basada en la coherencia.
                - "posterior_a_given_b": Probabilidad posterior de A dado B.
                - "conditional_action_given_b": Probabilidad condicional de la acción dado B.
        """
        return self.bayes_logic.calculate_probabilities_and_select_action(
            entropy, coherence, prn_influence, action
        )

    def perform_statistical_analysis(self, data: list[list[float]], point: list[float]
                                     ) -> tuple[np.ndarray, float]:
        """Realiza un análisis estadístico de un conjunto de datos

        Args:
            data (list[list[float]]): Matriz de datos donde cada fila representa una muestra y cada
                                  columna una característica.
            point (list[float]): Punto a evaluar

        Returns:
            tuple[np.ndarray, float]: Una tupla que contiene
            - la matriz de covarianza
            - la distancia de Mahalanobis entre el conjunto de datos y el `point`
        """

        covariance_matrix = self.stat_analysis.calculate_covariance_matrix(data)
        mahalanobis_distance = self.stat_analysis.compute_mahalanobis_distance(data, point)
        return covariance_matrix, mahalanobis_distance

    def get_fft_features(self, quantum_state: list[complex]) -> dict[str, np.ndarray | float]:
        """Obtiene las características FFT de un estado cuántico.

        Args:
            quantum_state (list[complex]): Estado cuántico representado como una lista de números complejos.

        Returns:
            dict[str, np.ndarray | float]: Diccionario con las características FFT calculadas:
                - "magnitudes": Magnitudes de los componentes de frecuencia.
                - "phases": Fases de los componentes de frecuencia.
                - "entropy": Entropía de Shannon de las magnitudes.
                - "coherence": Medida de coherencia basada en la varianza de las fases.
        """
        return self.fft_integrator.process_quantum_state(quantum_state)


# Ejemplo de uso (puedes integrarlo en tu interfaz)
if __name__ == "__main__":
    # Configura logging
    logging.basicConfig(level=logging.INFO)

    # 1. Crea una instancia del integrador
    integrator = AgentInterfaceIntegrator()

    # 2. Configura el agente
    agent_config = {
        'name': 'MyQuantumAgent',
        'num_qubits': 4,
        'learning_rate': 0.1,
        'input_size': 5,
        'hidden_size': 64,
        'rnn_type': 'GRU',
        'dropout_rate': 0.2,
        'use_batch_norm': True,
        'adaptative_lr_decay': 0.99
    }
    integrator.setup_agent(agent_config)

    # 3. Inicializa el circuito cuántico
    integrator.setup_quantum_circuit(num_qubits=4)

    # 4. Genera datos de ejemplo
    data = np.random.randn(1000, 5)

     # 5. Aplica algunas compuertas personalizadas (opcional)
    custom_gates = [
      ('h', None, [0]),       # Puerta Hadamard al qubit 0
      ('cx', None, [0, 1]),  # Puerta CNOT controlada en 0 y objetivo en 1
      ('rz', [np.pi/4], [2]) # Puerta RZ con fase pi/4 al qubit 2
    ]
    integrator.apply_custom_gates(custom_gates)

    # 5.1 Mide un qubit en específico
    integrator.measure_qubit(1)

    #5.2 Mide todos los qubits
    # integrator.measure_all_qubits()

    # 6. Obtén el estado cuántico
    qstate = integrator.get_quantum_state()
    print("Estado cuántico", qstate)

    # 7. Obtener probabilidades
    probs = integrator.get_probabilities()
    print("Probabilidades", probs)

    #8. Obtener características FFT
    fft_feats = integrator.get_fft_features(qstate)
    print("Características FFT", fft_feats)

    # 9. Calcula métricas bayesianas (necesitas proporcionar valores de ejemplo)
    entropy_example = 0.5
    coherence_example = 0.8
    prn_example = 0.6
    action_example = 1
    bayes_metrics = integrator.calculate_bayes_metrics(entropy_example, coherence_example, prn_example, action_example)
    print("Métricas bayesianas:", bayes_metrics)

    # 10. Realiza un análisis estadístico (datos y punto de ejemplo)
    data_example = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]
    point_example = [1.5, 2.5]
    covariance_matrix, mahalanobis_distance = integrator.perform_statistical_analysis(data_example, point_example)
    print("Matriz de Covarianza:", covariance_matrix)
    print("Distancia de Mahalanobis:", mahalanobis_distance)

    # 11. Obtén la configuración actual del agente
    current_settings = integrator.get_agent_settings()
    print("Configuración actual del agente:", current_settings)


    # 12. Ejecuta la simulación
    integrator.run_simulation(data, sequence_length=10, num_iterations=50, training_epochs=20, batch_size=32)

    # 13. Actualiza la configuración
    new_agent_config = {
        'num_qubits': 5,          # Increased qubits
        'learning_rate': 0.05,     # Reduced learning rate
        'hidden_size': 128,       # Increased hidden size
        'rnn_type': 'LSTM',       # Changed to LSTM
        'dropout_rate': 0.3,      # Increased dropout
        'use_batch_norm': False,   # Disabled batch normalization
        'adaptative_lr_decay': 0.95
    }
    integrator.update_agent_settings(new_agent_config)
    updated_settings = integrator.get_agent_settings()
    print("Configuración actualizada del agente:", updated_settings)

    # 14. Vuelve a ejecutar con la nueva configuración (opcional pero recomendado)
    integrator.run_simulation(data, sequence_length=10, num_iterations=50, training_epochs=20, batch_size=32)
    print("Simulación finalizada")
