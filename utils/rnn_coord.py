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