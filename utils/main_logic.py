import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Importar otros módulos
from cognitive_model import BayesLogic, StatisticalAnalysis
from quantum_state import QuantumState


class RNNCoordinator:
    """Coordinador para modelos RNN y regresión lineal con entrenamiento y predicción combinada."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Inicializa el coordinador RNN.
        
        Args:
            input_size: Dimensión de los datos de entrada
            hidden_size: Tamaño de las capas ocultas
            output_size: Dimensión de la salida
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.rnn_model = None
                
    def create_rnn_model(self) -> Sequential:
        """
        Crea un modelo RNN con LSTM y una capa de salida lineal.
        
        Returns:
            El modelo de Keras compilado
        """
        model = Sequential([
            SimpleRNN(self.hidden_size, input_shape=(None, self.input_size), 
                      return_sequences=True),
            LSTM(self.hidden_size//2),
            Dense(self.output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        self.rnn_model = model
        return model
        
    @staticmethod
    def prepare_data(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos secuenciales para entrenamiento RNN.
        
        Args:
            data: Datos de entrada
            sequence_length: Longitud de secuencia para el RNN
            
        Returns:
            Tupla de (X, y) preparados para entrenamiento
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     sequence_length: int, epochs: int = 100) -> dict:
        """
        Entrena ambos modelos: RNN y regresión lineal.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            sequence_length: Longitud de secuencia para RNN
            epochs: Número de épocas para el entrenamiento RNN
            
        Returns:
            Diccionario con el historial de entrenamiento
        """
        # Preparar datos secuenciales para RNN
        X_rnn, y_rnn = self.prepare_data(X_train, sequence_length)
                
        # Escalar los datos
        X_train_scaled = self.scaler.fit_transform(X_train)
                
        # Entrenar RNN
        if self.rnn_model is None:
            self.create_rnn_model()
        history = self.rnn_model.fit(
            X_rnn, y_rnn, 
            epochs=epochs, 
            verbose=1, 
            validation_split=0.2
        )
                
        # Entrenar regresión lineal
        self.linear_model.fit(X_train_scaled, y_train)
        
        return {'rnn_history': history.history}
        
    def save_models(self, base_filename: str) -> None:
        """
        Guarda ambos modelos y el scaler en archivos.
        
        Args:
            base_filename: Nombre base para los archivos
        """
        base_path = Path(base_filename)
        # Crear directorio si no existe
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo RNN en formato H5
        self.rnn_model.save(f'{base_filename}_rnn.h5')
                
        # Guardar modelo de regresión lineal y scaler usando pickle
        with open(f'{base_filename}_linear.pkl', 'wb') as f:
            pickle.dump({
                'linear_model': self.linear_model,
                'scaler': self.scaler
            }, f)
        
    def load_models(self, base_filename: str) -> bool:
        """
        Carga ambos modelos y el scaler desde archivos.
        
        Args:
            base_filename: Nombre base de los archivos
            
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            # Cargar modelo RNN
            self.rnn_model = load_model(f'{base_filename}_rnn.h5')
                    
            # Cargar modelo de regresión lineal y scaler
            with open(f'{base_filename}_linear.pkl', 'rb') as f:
                models_dict = pickle.load(f)
                self.linear_model = models_dict['linear_model']
                self.scaler = models_dict['scaler']
            return True
        except (FileNotFoundError, IOError) as e:
            print(f"Error al cargar los modelos: {e}")
            return False
        
    def predict(self, X: np.ndarray, sequence_length: int, rnn_weight: float = 0.7) -> np.ndarray:
        """
        Realiza predicciones combinando ambos modelos.
        
        Args:
            X: Datos de entrada
            sequence_length: Longitud de secuencia para RNN
            rnn_weight: Peso para las predicciones RNN (entre 0 y 1)
            
        Returns:
            Predicciones combinadas
        """
        if self.rnn_model is None:
            raise ValueError("El modelo RNN no ha sido entrenado o cargado aún")
            
        X_rnn, _ = self.prepare_data(X, sequence_length)
        X_scaled = self.scaler.transform(X)
        
        rnn_pred = self.rnn_model.predict(X_rnn)
        linear_pred = self.linear_model.predict(X_scaled)
        
        # Asegurar que las predicciones tengan la misma longitud
        lin_pred_adjusted = linear_pred[sequence_length:]
        
        # Verificar dimensiones
        if len(rnn_pred) != len(lin_pred_adjusted):
            raise ValueError(f"Dimensiones incompatibles: RNN ({len(rnn_pred)}) vs Linear ({len(lin_pred_adjusted)})")
            
        combined_pred = (rnn_weight * rnn_pred + (1 - rnn_weight) * lin_pred_adjusted)
        return combined_pred


class QuantumLearningAgent:
    """Agente de aprendizaje basado en estados cuánticos."""
    
    def __init__(self, name: str, num_qubits: int = 4, learning_rate: float = 0.1):
        """
        Inicializa el agente de aprendizaje cuántico.
        
        Args:
            name: Nombre del agente
            num_qubits: Número de qubits para el estado cuántico
            learning_rate: Tasa de aprendizaje
        """
        self.name = name
        num_positions = 2**num_qubits
        self.quantum_state = QuantumState(num_positions, learning_rate)
        self.memory: List[Dict[str, Any]] = []
        self.total_reward = 0
        self.learning_rate = learning_rate
        self.exploration_rate = 0.1
        self._last_observation: Optional[int] = None

    def quantum_observe(self) -> int:
        """
        Observa el estado cuántico actual.
        
        Returns:
            Posición observada
        """
        self._last_observation = self.quantum_state.observe_position()
        return self._last_observation

    def update_quantum_state(self, action: int) -> None:
        """
        Actualiza el estado cuántico basado en una acción.
        
        Args:
            action: Índice de la acción tomada
        """
        self.quantum_state.update_probabilities(action)

    def learn_from_experience(self, experience: Dict[str, Any]) -> float:
        """
        Aprende de una experiencia y actualiza el estado interno.
        
        Args:
            experience: Diccionario con los datos de la experiencia
            
        Returns:
            Recompensa recibida
        """
        self.memory.append(experience)
        reward = experience.get('reward', 0)
        self.total_reward += reward
        action = experience.get('action', 0)
        self.update_quantum_state(action)
        return reward

    def choose_action(self, state: Dict[str, Any]) -> int:
        """
        Elige una acción basada en el estado actual.
        
        Args:
            state: Estado actual
            
        Returns:
            Acción elegida
        """
        # Exploración vs explotación
        if random.random() < self.exploration_rate:
            return random.randint(0, 1)
            
        # Decisión basada en probabilidades cuánticas
        probabilities = self.quantum_state.probabilities
        return int(np.argmax(probabilities))

    def simulate_interaction(self, num_iterations: int = 50) -> Dict[str, List[float]]:
        """
        Simula interacciones y aprendizaje.
        
        Args:
            num_iterations: Número de iteraciones para la simulación
            
        Returns:
            Diccionario con datos de la simulación
        """
        print(f"Iniciando simulación para {self.name}")
        
        results = {
            'rewards': [],
            'actions': [],
            'probabilities': []
        }
        
        for iteration in range(num_iterations):
            current_state = {
                'iteration': iteration,
                'random_factor': random.random()
            }
            
            action = self.choose_action(current_state)
            reward = self._calculate_reward(action, current_state)
            
            experience = {
                'state': current_state,
                'action': action,
                'reward': reward
            }
            
            self.learn_from_experience(experience)
            
            # Guardar resultados para análisis
            results['rewards'].append(reward)
            results['actions'].append(action)
            results['probabilities'].append(self.quantum_state.probabilities.copy())
            
        # Visualizar evolución
        self.quantum_state.visualize_state_evolution(save_path=f'{self.name}_quantum_evolution.png')
        
        return results

    def _calculate_reward(self, action: int, state: Dict[str, Any]) -> float:
        """
        Calcula la recompensa para una acción en un estado dado.
        
        Args:
            action: Acción tomada
            state: Estado actual
            
        Returns:
            Valor de recompensa
        """
        base_reward = state['random_factor']
        action_modifier = 1.0 if action == 1 else -0.5
        return base_reward * action_modifier


class MainLogic:
    """Lógica principal que integra todos los componentes."""
    
    def __init__(self, rnn_config: Dict[str, int] = None, quantum_config: Dict[str, Any] = None):
        """
        Inicializa la lógica principal.
        
        Args:
            rnn_config: Configuración para el RNNCoordinator
            quantum_config: Configuración para el QuantumLearningAgent
        """
        # Configuraciones por defecto
        rnn_default = {'input_size': 5, 'hidden_size': 64, 'output_size': 5}
        quantum_default = {'name': "QuantumExplorer", 'num_qubits': 4, 'learning_rate': 0.1}
        
        # Usar configuraciones proporcionadas o valores por defecto
        rnn_params = {**rnn_default, **(rnn_config or {})}
        quantum_params = {**quantum_default, **(quantum_config or {})}
        
        # Inicializar componentes
        self.bayes = BayesLogic()
        self.stats = StatisticalAnalysis()
        self.rnn_coordinator = RNNCoordinator(**rnn_params)
        self.quantum_agent = QuantumLearningAgent(**quantum_params)
        
        # Rastreo de datos de entrenamiento y resultados
        self.training_results = {}
        self.prediction_results = {}
        self.quantum_results = {}

    def process_data(self, data: np.ndarray) -> Dict[str, float]:
        """
        Procesa datos con lógica bayesiana y análisis estadístico.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        results = {}
        
        # Calcular entropía
        results['entropy'] = self.stats.shannon_entropy(data)
        
        # Análisis bayesiano (ejemplo)
        if hasattr(self.bayes, 'calculate_posterior'):
            results['posterior'] = self.bayes.calculate_posterior(data)
            
        # Añadir estadísticas básicas
        results['mean'] = np.mean(data)
        results['std'] = np.std(data)
        results['min'] = np.min(data)
        results['max'] = np.max(data)
        
        return results

    def train_and_predict(self, data: np.ndarray, sequence_length: int, epochs: int = 50,
                          test_size: float = 0.2) -> Dict[str, Any]:
        """
        Entrena modelos y realiza predicciones.
        
        Args:
            data: Conjunto completo de datos
            sequence_length: Longitud de secuencia para RNN
            epochs: Número de épocas para entrenamiento
            test_size: Proporción de datos para prueba
            
        Returns:
            Diccionario con resultados y métricas
        """
        # Validación de datos
        if len(data) <= sequence_length:
            raise ValueError(f"Los datos deben tener más elementos que sequence_length ({sequence_length})")
            
        # División en conjuntos de entrenamiento y prueba
        split_idx = int((1 - test_size) * len(data))
        X_train, y_train = data[:split_idx], data[1:split_idx+1]
        X_test = data[split_idx:]
        y_test = data[split_idx+1:] if split_idx+1 < len(data) else []
        
        # Entrenamiento
        training_history = self.rnn_coordinator.train_models(X_train, y_train, sequence_length, epochs)
        
        # Predicción
        if len(X_test) > sequence_length:
            predictions = self.rnn_coordinator.predict(X_test, sequence_length)
            
            # Cálculo de métricas si hay datos de prueba suficientes
            metrics = {}
            if len(y_test) >= len(predictions):
                y_true = y_test[:len(predictions)]
                mse = np.mean((predictions - y_true) ** 2)
                mae = np.mean(np.abs(predictions - y_true))
                metrics = {'mse': mse, 'mae': mae}
        else:
            predictions = np.array([])
            metrics = {'error': 'Datos de prueba insuficientes'}
            
        # Guardar resultados
        results = {
            'training_history': training_history,
            'predictions': predictions,
            'metrics': metrics,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        self.training_results = results
        return results

    def quantum_interaction(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Ejecuta la interacción cuántica del agente.
        
        Args:
            iterations: Número de iteraciones para la simulación
            
        Returns:
            Resultados de la interacción cuántica
        """
        results = self.quantum_agent.simulate_interaction(iterations)
        self.quantum_results = results
        return results
        
    def save_all_models(self, base_path: str) -> None:
        """
        Guarda todos los modelos y estados.
        
        Args:
            base_path: Directorio base para guardar
        """
        # Crear directorio si no existe
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        # Guardar modelos RNN
        self.rnn_coordinator.save_models(f"{base_path}/models")
        
        # Guardar resultados y configuración
        with open(f"{base_path}/results.pkl", 'wb') as f:
            pickle.dump({
                'training_results': self.training_results,
                'quantum_results': self.quantum_results,
                'prediction_results': self.prediction_results
            }, f)
        
    def load_all_models(self, base_path: str) -> bool:
        """
        Carga todos los modelos y estados.
        
        Args:
            base_path: Directorio base para cargar
            
        Returns:
            True si la carga fue exitosa
        """
        try:
            # Cargar modelos RNN
            rnn_loaded = self.rnn_coordinator.load_models(f"{base_path}/models")
            
            # Cargar resultados
            with open(f"{base_path}/results.pkl", 'rb') as f:
                results = pickle.load(f)
                self.training_results = results.get('training_results', {})
                self.quantum_results = results.get('quantum_results', {})
                self.prediction_results = results.get('prediction_results', {})
                
            return rnn_loaded
        except Exception as e:
            print(f"Error al cargar los modelos y estados: {e}")
            return False

# ejemplo de uso 
logic = MainLogic()
data = np.random.rand(100, 5)  # Datos de ejemplo
results = logic.train_and_predict(data, sequence_length=10)
quantum_results = logic.quantum_interaction()
logic.save_all_models("resultados")
