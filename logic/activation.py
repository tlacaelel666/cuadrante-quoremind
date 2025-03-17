import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
from enum import Enum

class QuantumNeuralHybridSystem:
    """
    Sistema Neural Híbrido Cuántico-Clásico Avanzado
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_layers: List[int], 
        output_size: int,
        quantum_layers: int = 3,
        learning_rate: float = 0.01
    ):
        # Componentes del sistema
        self.classical_network = NeuralNetwork(
            input_size, hidden_layers, output_size
        )
        self.quantum_activator = QuantumClassicalActivator(
            n_qubits=quantum_layers
        )
        self.quantum_state = QuantumState(
            num_positions=output_size, 
            learning_rate=learning_rate
        )
        
        # Parámetros de entrenamiento
        self.learning_rate = learning_rate
        self.quantum_memory = []
    
    def hybrid_forward_propagation(
        self, 
        X: np.ndarray, 
        activation_type: ActivationType = ActivationType.SIGMOID
    ) -> Tuple[np.ndarray, float]:
        """
        Propagación forward híbrida que combina red neuronal clásica 
        con activación cuántica
        """
        # Propagación clásica
        classical_activations = self.classical_network.forward(X)
        final_layer = classical_activations[-1]
        
        # Activación cuántica
        quantum_activated_output, collapse_prob = self.quantum_activator.quantum_activated_forward(
            final_layer, 
            activation_type
        )
        
        # Actualizar estado cuántico
        self.quantum_state.update_state(
            action=1 if collapse_prob > 0.5 else 0
        )
        
        # Registrar en memoria cuántica
        self.quantum_memory.append({
            'input': X,
            'classical_output': final_layer,
            'quantum_output': quantum_activated_output,
            'collapse_probability': collapse_prob
        })
        
        return quantum_activated_output, collapse_prob
    
    def hybrid_backpropagation(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        activation_type: ActivationType = ActivationType.SIGMOID
    ):
        """
        Retropropagación híbrida que integra gradientes clásicos y cuánticos
        """
        # Propagación forward
        quantum_output, collapse_prob = self.hybrid_forward_propagation(X, activation_type)
        
        # Error de salida
        output_error = y - quantum_output
        
        # Gradiente híbrido
        hybrid_gradient = self.quantum_activator.hybrid_backpropagation(
            output_error, 
            activation_type, 
            collapse_prob
        )
        
        # Retropropagación clásica con gradiente modificado
        self.classical_network.backward(
            X, 
            hybrid_gradient, 
            learning_rate=self.learning_rate,
            optimizer='adam'
        )
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Entrenamiento del sistema híbrido
        """
        for epoch in range(epochs):
            # Entrenamiento por lotes
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Retropropagación híbrida
                self.hybrid_backpropagation(X_batch, y_batch)
            
            # Monitoreo de entrenamiento
            if epoch % 10 == 0:
                self._log_training_progress(epoch)
    
    def _log_training_progress(self, epoch: int):
        """
        Registro de progreso de entrenamiento
        """
        # Análisis de memoria cuántica
        if self.quantum_memory:
            recent_memory = self.quantum_memory[-10:]
            avg_collapse_prob = np.mean([
                entry['collapse_probability'] for entry in recent_memory
            ])
            
            print(f"Época {epoch}:")
            print(f"  Probabilidad promedio de colapso: {avg_collapse_prob:.4f}")
            print(f"  Estado cuántico actual:")
            self.quantum_state.visualize_state()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicción con sistema híbrido
        """
        quantum_output, _ = self.hybrid_forward_propagation(X)
        return quantum_output
    
    def save_model(self, filepath: str):
        """
        Guardar modelo híbrido
        """
        import joblib
        
        model_data = {
            'classical_weights': self.classical_network.weights,
            'classical_biases': self.classical_network.biases,
            'quantum_state': self.quantum_state.state_vector,
            'quantum_memory': self.quantum_memory
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Cargar modelo híbrido
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.classical_network.weights = model_data['classical_weights']
        self.classical_network.biases = model_data['classical_biases']
        self.quantum_state.state_vector = model_data['quantum_state']
        self.quantum_memory = model_data['quantum_memory']

# Ejemplo de uso
def quantum_neural_hybrid_demo():
    # Datos de ejemplo
    X_train = np.random.rand(100, 5)
    y_train = np.sin(X_train.sum(axis=1)).reshape(-1, 1)
    
    # Crear sistema híbrido
    hybrid_system = QuantumNeuralHybridSystem(
        input_size=5, 
        hidden_layers=[10, 7], 
        output_size=1,
        quantum_layers=3
    )
    
    # Entrenar
    hybrid_system.train(X_train, y_train, epochs=50)
    
    # Predecir
    X_test = np.random.rand(10, 5)
    predictions = hybrid_system.predict(X_test)
    
    print("Predicciones finales:")
    print(predictions)
    
    # Guardar modelo
    hybrid_system.save_model('quantum_hybrid_model.joblib')

# Ejecutar demo
if __name__ == "__main__":
    quantum_neural_hybrid_demo()