"""
Módulo de Aprendizaje Cuántico Avanzado

Este módulo integra conceptos de estados cuánticos, RNN y aprendizaje por refuerzo.

Componentes principales:
- QuantumState: Simulación de estado cuántico
- RNNCoordinator: Coordinación de modelos de aprendizaje
- QuantumLearningAgent: Agente de aprendizaje con componente cuántico

Autor: Jacobo Tlacaelel Mina Rodríguez 
Fecha: 13/03/2025
Versión: cuadrante-coremind v1.0
"""
# rnn_coord.py

import numpy as np
import random
import pickle
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# Se importa la clase QuantumBayesMahalanobis (asegúrate de que esté en el PATH o en el mismo directorio)
from quantum_bayes_mahalanobis import QuantumBayesMahalanobis
from bayes_logic import StatisticalAnalysis  # Se asume que StatisticalAnalysis está en bayes_logic.py

# Clase QuantumState

class QuantumState(QuantumBayesMahalanobis):
    """
    Simula un estado cuántico utilizando la lógica bayesiana y cálculos de Mahalanobis.
    Hereda de QuantumBayesMahalanobis para disponer de métodos como:
       - predict_quantum_state
       - quantum_cosine_projection
       - calculate_quantum_posterior_with_mahalanobis
       
    Además, gestiona un vector de estado (probabilidades) que se normaliza y se actualiza
    en función de las acciones ejecutadas.
    """
    def __init__(self, num_positions: int, learning_rate: float = 0.1):
        """
        Inicializa el estado cuántico.
        
        Args:
            num_positions (int): Número de posiciones (o dimensión) del estado.
            learning_rate (float): Tasa de aprendizaje para actualizar el estado.
        """
        super().__init__()  # Inicializa QuantumBayesMahalanobis
        self.num_positions = num_positions
        self.learning_rate = learning_rate
        # Se inicializa el estado aleatoriamente y se normaliza para representar un vector de probabilidades
        self.state_vector = np.random.rand(num_positions)
        self.state_vector = self.normalize_state(self.state_vector)
        # Guardamos las probabilidades en un atributo (para la selección de acciones, por ejemplo)
        self.probabilities = self.state_vector.copy()
    
    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        """
        Normaliza el vector de estado para que su norma sea 1.
        
        Args:
            state (np.ndarray): Vector de estado.
            
        Returns:
            np.ndarray: Estado normalizado.
        """
        norm = np.linalg.norm(state)
        return state / norm if norm != 0 else state

    def predict_quantum_state_update(self) -> Tuple[np.ndarray, float]:
        """
        Usa el método predict_quantum_state (heredado de QuantumBayesMahalanobis)
        para predecir el siguiente estado cuántico basándose en el estado actual.
        
        Se calcula una especie de “posterior” cuántico que se utilizará para actualizar el estado.
        
        Returns:
            Tuple[np.ndarray, float]: (nuevo estado, posterior)
        """
        # Para calcular medidas cuánticas se asume que el input es de forma (1, num_features).
        # Aquí usamos el estado actual como entrada. Se establecen medidas simples de entropía y coherencia.
        # La entropía se deriva con StatisticalAnalysis.shannon_entropy y la coherencia se estima con la media.
        entropy = StatisticalAnalysis.shannon_entropy(self.state_vector)
        coherence = np.mean(self.state_vector)
        
        # predict_quantum_state retorna (next_state, posterior)
        next_state_tensor, posterior = self.predict_quantum_state(
            np.array([self.state_vector]), 
            entropy, 
            coherence
        )
        # Convertir el tensor a array
        next_state = next_state_tensor.numpy().flatten()
        next_state = self.normalize_state(next_state)
        return next_state, posterior

    def update_probabilities(self, action: int) -> None:
        """
        Actualiza el estado cuántico en función de la acción tomada.
        Esta función simula un “colapso” o actualización del estado a partir de la 
        predicción cuántica.
        
        Args:
            action (int): Acción tomada (por ejemplo, 0 o 1).
        """
        # Se predice el nuevo estado usando la función integrada de QuantumBayesMahalanobis
        next_state, posterior = self.predict_quantum_state_update()
        # Se utiliza la tasa de aprendizaje para actualizar de forma gradual
        self.state_vector = self.normalize_state(
            (1 - self.learning_rate) * self.state_vector +
            self.learning_rate * next_state
        )
        self.probabilities = self.state_vector.copy()

    def visualize_state_evolution(self):
        """
        Imprime en consola el estado cuántico actual.
        """
        print("Estado cuántico final (vector de probabilidades):")
        print(self.state_vector)

# Clase RNNCoordinator

class RNNCoordinator:
    """
    Coordinador para modelos RNN y regresión lineal.
    
    Gestiona el entrenamiento, predicción y el guardado de modelos híbridos.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Inicializa el coordinador con parámetros de modelo.
        
        Args:
            input_size (int): Tamaño de entrada.
            hidden_size (int): Tamaño de la capa oculta.
            output_size (int): Tamaño de salida.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.rnn_model = None
        
    def create_rnn_model(self) -> tf.keras.Model:
        """
        Crea y compila un modelo RNN con una capa LSTM y salida lineal.
        
        Returns:
            tf.keras.Model: Modelo RNN compilado.
        """
        model = Sequential([
            SimpleRNN(self.hidden_size, 
                      input_shape=(None, self.input_size), 
                      return_sequences=True),
            LSTM(self.hidden_size // 2),
            Dense(self.output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        self.rnn_model = model
        return model
    
    def prepare_data(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos secuenciales para el entrenamiento de la RNN.
        
        Args:
            data (np.ndarray): Datos de entrada.
            sequence_length (int): Longitud de la secuencia.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Datos X e y preparados.
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     sequence_length: int, epochs: int = 100) -> None:
        """
        Entrena los modelos RNN y la regresión lineal.
        
        Args:
            X_train (np.ndarray): Datos de entrenamiento.
            y_train (np.ndarray): Etiquetas de entrenamiento.
            sequence_length (int): Longitud de la secuencia.
            epochs (int, optional): Número de épocas de entrenamiento. Defaults to 100.
        """
        # Preparar datos para RNN
        X_rnn, y_rnn = self.prepare_data(X_train, sequence_length)
        
        # Escalar datos (se escala la totalidad de X_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Entrenar la RNN
        if self.rnn_model is None:
            self.create_rnn_model()
        self.rnn_model.fit(X_rnn, y_rnn, epochs=epochs, verbose=1)
        
        # Entrenar la regresión lineal
        self.linear_model.fit(X_train_scaled, y_train)
    
    def predict(self, X: np.ndarray, sequence_length: int, rnn_weight: float = 0.7) -> np.ndarray:
        """
        Realiza predicciones combinando los modelos RNN y de regresión lineal.
        
        Args:
            X (np.ndarray): Datos de entrada.
            sequence_length (int): Longitud de la secuencia.
            rnn_weight (float, optional): Peso del modelo RNN en la combinación. Defaults to 0.7.
        
        Returns:
            np.ndarray: Predicciones combinadas.
        """
        X_rnn, _ = self.prepare_data(X, sequence_length)
        X_scaled = self.scaler.transform(X)
        
        rnn_pred = self.rnn_model.predict(X_rnn)
        # Para la regresión lineal, se descartan las primeras "sequence_length" predicciones
        linear_pred = self.linear_model.predict(X_scaled)[sequence_length:]
        
        combined_pred = rnn_weight * rnn_pred + (1 - rnn_weight) * linear_pred
        return combined_pred


# Clase QuantumLearningAgent

class QuantumLearningAgent:
    """
    Agente de aprendizaje que integra conceptos de estados cuánticos y aprendizaje por refuerzo.
    Combina un estado cuántico avanzado (QuantumState) y un coordinador RNN para establecer
    una política híbrida.
    """
    def __init__(self, 
                 name: str, 
                 num_qubits: int = 4, 
                 learning_rate: float = 0.1,
                 input_size: int = 5,
                 hidden_size: int = 64):
        """
        Inicializa el agente de aprendizaje cuántico.
        
        Args:
            name (str): Nombre del agente.
            num_qubits (int, optional): Número de qubits para definir el tamaño del estado cuántico.
            learning_rate (float, optional): Tasa de aprendizaje para actualizar el estado.
            input_size (int, optional): Tamaño de la entrada para la RNN.
            hidden_size (int, optional): Tamaño de la capa oculta de la RNN.
        """
        self.name = name
        # Se define el número de posiciones como 2^num_qubits
        num_positions = 2 ** num_qubits
        self.quantum_state = QuantumState(num_positions, learning_rate)
        self.rnn_coordinator = RNNCoordinator(input_size, hidden_size, output_size=input_size)
        self.memory: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        
        # Parámetros de exploración para aprendizaje por refuerzo
        self.learning_rate = learning_rate
        self.exploration_rate = 0.1

    def simulate_quantum_rnn_interaction(self, 
                                         data: np.ndarray, 
                                         sequence_length: int = 10, 
                                         num_iterations: int = 50) -> None:
        """
        Simula interacciones combinando el estado cuántico y el coordinador RNN.
        
        Args:
            data (np.ndarray): Datos de entrada.
            sequence_length (int, optional): Longitud de secuencia para entrenamiento y predicción.
            num_iterations (int, optional): Número de iteraciones (episodios) de interacción.
        """
        # Se divide el conjunto de datos para entrenamiento y prueba
        split_idx = int(0.8 * len(data))
        X_train, y_train = data[:split_idx], data[1:split_idx+1]
        X_test = data[split_idx:]
        
        # Entrenar los modelos del coordinador RNN
        self.rnn_coordinator.train_models(X_train, y_train, sequence_length, epochs=50)
        
        # Simular iteraciones de interacción
        for iteration in range(num_iterations):
            # Elegir acción basada en la probabilidad del estado cuántico
            action = self.choose_action()
            
            # Obtener predicciones usando el modelo híbrido
            predictions = self.rnn_coordinator.predict(X_test, sequence_length)
            
            # Calcular la recompensa (ejemplo: basada en la varianza de las predicciones y la acción)
            reward = self._calculate_reward(predictions, action)
            
            # Actualizar el estado cuántico a partir de la acción ejecutada
            self.quantum_state.update_probabilities(action)
            
            # Registrar la experiencia
            self.memory.append({
                'iteration': iteration,
                'action': action,
                'reward': reward,
                'predictions': predictions
            })
            self.total_reward += reward
        
        # Visualizar la evolución final del estado cuántico
        self.quantum_state.visualize_state_evolution()

    def choose_action(self) -> int:
        """
        Selecciona una acción en base a la política de exploración o explotación.
        
        Returns:
            int: Acción elegida (por ejemplo, 0 o 1).
        """
        if random.random() < self.exploration_rate:
            return random.randint(0, 1)
        # Se asume que self.quantum_state.probabilities es un vector; se elige la acción dominante
        return int(np.argmax(self.quantum_state.probabilities))

    def _calculate_reward(self, predictions: np.ndarray, action: int) -> float:
        """
        Calcula la recompensa en función de las predicciones y la acción tomada.
        
        Args:
            predictions (np.ndarray): Predicciones del modelo.
            action (int): Acción ejecutada.
        
        Returns:
            float: Valor de recompensa obtenido.
        """
        # Ejemplo de cálculo: se usa la varianza de las predicciones y un modificador según la acción
        prediction_variance = np.var(predictions)
        action_modifier = 1.0 if action == 1 else -0.5
        return prediction_variance * action_modifier


# Función principal de demostración

def main():
    """
    Función principal para demostrar el agente de aprendizaje cuántico-RNN.
    Se generan datos de ejemplo, se simulan interacciones y se muestra un resumen.
    """
    # Generar datos de ejemplo
    np.random.seed(42)
    time_steps = 1000
    input_size = 5
    data = np.random.randn(time_steps, input_size)
    
    # Crear agente de aprendizaje cuántico
    quantum_agent = QuantumLearningAgent(
        name="QuantumRNNExplorer", 
        num_qubits=4,
        learning_rate=0.1,
        input_size=input_size,
        hidden_size=64
    )
    
    # Simular interacciones del agente
    quantum_agent.simulate_quantum_rnn_interaction(data, sequence_length=10, num_iterations=100)
    
    # Imprimir resumen de la simulación
    print(f"\nResumen del Agente {quantum_agent.name}")
    print(f"Recompensa Total: {quantum_agent.total_reward}")
    print(f"Número de Experiencias Registradas: {len(quantum_agent.memory)}")

if __name__ == "__main__":
    main()