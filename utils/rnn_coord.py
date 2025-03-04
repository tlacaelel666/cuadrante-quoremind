class RNNCoordinator:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.rnn_model = None
        
    def create_rnn_model(self):
        """Creates a simple RNN model with linear regression output layer"""
        model = Sequential([
            SimpleRNN(self.hidden_size, input_shape=(None, self.input_size), 
                     return_sequences=True),
            LSTM(self.hidden_size//2),
            Dense(self.output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        self.rnn_model = model
        return model
    
    def prepare_data(self, data, sequence_length):
        """Prepares sequential data for RNN training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train_models(self, X_train, y_train, sequence_length, epochs=100):
        """Trains both RNN and linear regression models"""
        # Prepare sequential data for RNN
        X_rnn, y_rnn = self.prepare_data(X_train, sequence_length)
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train RNN
        if self.rnn_model is None:
            self.create_rnn_model()
        self.rnn_model.fit(X_rnn, y_rnn, epochs=epochs, verbose=1)
        
        # Train linear regression
        self.linear_model.fit(X_train_scaled, y_train)
    
    def save_models(self, base_filename):
        """Saves both models and scaler to files"""
        # Save RNN model in H5 format
        self.rnn_model.save(f'{base_filename}_rnn.h5')
        
        # Save linear regression model and scaler using pickle
        with open(f'{base_filename}_linear.pkl', 'wb') as f:
            pickle.dump({
                'linear_model': self.linear_model,
                'scaler': self.scaler
            }, f)
    
    def load_models(self, base_filename):
        """Loads both models and scaler from files"""
        # Load RNN model
        self.rnn_model = tf.keras.models.load_model(f'{base_filename}_rnn.h5')
        
        # Load linear regression model and scaler
        with open(f'{base_filename}_linear.pkl', 'rb') as f:
            models_dict = pickle.load(f)
            self.linear_model = models_dict['linear_model']
            self.scaler = models_dict['scaler']
    
    def predict(self, X, sequence_length):
        """Makes predictions using both models and combines them"""
        # Prepare data for RNN prediction
        X_rnn, _ = self.prepare_data(X, sequence_length)
        
        # Scale data for linear regression
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rnn_pred = self.rnn_model.predict(X_rnn)
        linear_pred = self.linear_model.predict(X_scaled)
        
        # Combine predictions (using average as an example)
        combined_pred = (rnn_pred + linear_pred[sequence_length:]) / 2
        return combined_pred

# Example usage
def example_usage():
    # Sample data generation
    np.random.seed(42)
    time_steps = 1000
    input_size = 5
    data = np.random.randn(time_steps, input_size)
    
    # Initialize coordinator
    coordinator = RNNCoordinator(
        input_size=input_size,
        hidden_size=64,
        output_size=input_size
    )
    
    # Train models
    sequence_length = 10
    split_idx = int(0.8 * len(data))
    X_train, y_train = data[:split_idx], data[1:split_idx+1]
    coordinator.train_models(X_train, y_train, sequence_length, epochs=50)
    
    # Save models
    coordinator.save_models('model_files')
    
    # Load models
    new_coordinator = RNNCoordinator(input_size, 64, input_size)
    new_coordinator.load_models('model_files')
    
    # Make predictions
    X_test = data[split_idx:]
    predictions = new_coordinator.predict(X_test, sequence_length)
    
    return predictions

if __name__ == "__main__":
    predictions = example_usage()
    print("Predictions shape:", predictions.shape))


def predict(self, X, sequence_length, rnn_weight=0.7):
    """Makes predictions using both models and combines them with weights"""
    X_rnn, _ = self.prepare_data(X, sequence_length)
    X_scaled = self.scaler.transform(X)
    rnn_pred = self.rnn_model.predict(X_rnn)
    linear_pred = self.linear_model.predict(X_scaled)
    combined_pred = (rnn_weight * rnn_pred + (1 - rnn_weight) * linear_pred[sequence_length:])
    return combined_pred

#aprenfizaje cuántico 
import numpy as np
import ast
import random
from typing import Dict, Any, List

class QuantumLearningAgent:
    """
    Agente que combina conceptos de estado cuántico con estrategias de aprendizaje.
    """
    def __init__(
        self, 
        name: str, 
        num_qubits: int = 4, 
        learning_rate: float = 0.1
    ):
        """
        Inicializa un agente de aprendizaje cuántico.

        Args:
            name (str): Nombre del agente.
            num_qubits (int): Número de qubits para el estado cuántico.
            learning_rate (float): Tasa de aprendizaje.
        """
        self.name = name
        
        # Estado cuántico como base para el aprendizaje
        num_positions = 2**num_qubits
        self.quantum_state = QuantumState(num_positions, learning_rate)
        
        # Componentes de aprendizaje
        self.memory: List[Dict[str, Any]] = []
        self.total_reward = 0
        
        # Configuración de aprendizaje
        self.learning_rate = learning_rate
        self.exploration_rate = 0.1

    def quantum_observe(self) -> int:
        """
        Observa el estado cuántico utilizando probabilidades.

        Returns:
            int: Posición observada
        """
        return self.quantum_state.observe_position()

    def update_quantum_state(self, action: int) -> None:
        """
        Actualiza el estado cuántico basado en una acción.

        Args:
            action (int): Acción tomada (0 o 1)
        """
        self.quantum_state.update_probabilities(action)

    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """
        Aprende de una experiencia almacenada.

        Args:
            experience (Dict[str, Any]): Diccionario de experiencia.
        """
        # Almacenar experiencia
        self.memory.append(experience)
        
        # Simular actualización basada en recompensa
        reward = experience.get('reward', 0)
        self.total_reward += reward
        
        # Actualizar estado cuántico
        action = experience.get('action', 0)
        self.update_quantum_state(action)

    def choose_action(self, state: Dict[str, Any]) -> int:
        """
        Elige una acción utilizando estrategia epsilon-greedy con estado cuántico.

        Args:
            state (Dict[str, Any]): Estado actual del entorno.

        Returns:
            int: Acción elegida (0 o 1)
        """
        # Exploración basada en probabilidades cuánticas
        if random.random() < self.exploration_rate:
            return random.randint(0, 1)
        
        # Explotación usando probabilidades del estado cuántico
        probabilities = self.quantum_state.probabilities
        return np.argmax(probabilities)

    def simulate_interaction(self, num_iterations: int = 50) -> None:
        """
        Simula una serie de interacciones e iteraciones de aprendizaje.

        Args:
            num_iterations (int): Número de iteraciones de simulación.
        """
        print(f"Iniciando simulación para {self.name}")
        
        for iteration in range(num_iterations):
            # Simular estado
            current_state = {
                'iteration': iteration,
                'random_factor': random.random()
            }
            
            # Elegir acción
            action = self.choose_action(current_state)
            
            # Simular recompensa (lógica simplificada)
            reward = self._calculate_reward(action, current_state)
            
            # Crear experiencia
            experience = {
                'state': current_state,
                'action': action,
                'reward': reward
            }
            
            # Aprender de la experiencia
            self.learn_from_experience(experience)
        
        # Visualizar resultados
        self.quantum_state.visualize_state_evolution(
            save_path=f'{self.name}_quantum_evolution.png'
        )

    def _calculate_reward(self, action: int, state: Dict[str, Any]) -> float:
        """
        Calcula una recompensa simulada basada en la acción y el estado.

        Args:
            action (int): Acción tomada
            state (Dict[str, Any]): Estado actual

        Returns:
            float: Recompensa calculada
        """
        # Lógica de recompensa basada en características del estado
        base_reward = state['random_factor']
        action_modifier = 1 if action == 1 else -0.5
        
        return base_reward * action_modifier

def main():
    """Función principal para demostrar el agente de aprendizaje cuántico."""
    # Crear agente de aprendizaje cuántico
    quantum_agent = QuantumLearningAgent(
        name="QuantumLearningExplorer", 
        num_qubits=4,
        learning_rate=0.1
    )
    
    # Simular interacciones
    quantum_agent.simulate_interaction(num_iterations=100)
    
    # Imprimir resumen
    print(f"\nResumen del Agente {quantum_agent.name}")
    print(f"Recompensa Total: {quantum_agent.total_reward}")
    print(f"Número de Experiencias Aprendidas: {len(quantum_agent.memory)}")

if __name__ == "__main__":
    main()