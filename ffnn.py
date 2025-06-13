"""
Fully Connected Feedforward Neural Network (FC-FFNN) Implementation
Core neural network class with configurable architecture
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import pickle

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x
    
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class FCFFNN:
    """
    Fully Connected Feedforward Neural Network
    
    Implements a configurable neural network with multiple hidden layers
    for function approximation and classification tasks.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 activation: str = 'relu',
                 output_activation: str = 'linear',
                 random_seed: Optional[int] = None):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        
        # Get activation functions
        self.activation_func = getattr(ActivationFunctions, activation)
        self.activation_derivative = getattr(ActivationFunctions, f"{activation}_derivative")
        self.output_activation_func = getattr(ActivationFunctions, output_activation)
        self.output_activation_derivative = getattr(ActivationFunctions, f"{output_activation}_derivative")
        
        # Initialize network architecture
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        # Training history
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
    
    def _initialize_parameters(self):
        """Initialize network weights and biases using Xavier initialization"""
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            # Weight initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weight = np.random.uniform(-limit, limit, (fan_in, fan_out))
            self.weights.append(weight)
            
            # Bias initialization (small random values)
            bias = np.random.uniform(-0.1, 0.1, (1, fan_out))
            self.biases.append(bias)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            output: Network output
            activations: List of activations for each layer
            z_values: List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        current_input = X
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            a = self.activation_func(z)
            activations.append(a)
            current_input = a
        
        # Output layer
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        
        output = self.output_activation_func(z_output)
        activations.append(output)
        
        return output, activations, z_values
    
    def backward(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 activations: List[np.ndarray], 
                 z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients
        
        Args:
            X: Input data
            y: Target values
            activations: Activations from forward pass
            z_values: Pre-activation values from forward pass
            
        Returns:
            weight_gradients: List of weight gradients
            bias_gradients: List of bias gradients
        """
        m = X.shape[0]  # batch size
        
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error
        delta = (activations[-1] - y) * self.output_activation_derivative(z_values[-1])
        
        # Compute gradients for each layer (backwards)
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Compute delta for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, 
                         weight_gradients: List[np.ndarray], 
                         bias_gradients: List[np.ndarray], 
                         learning_rate: float,
                         l2_lambda: float = 0.0):
        """Update network parameters using gradients"""
        for i in range(len(self.weights)):
            # L2 regularization
            l2_penalty = l2_lambda * self.weights[i]
            
            # Update weights and biases
            self.weights[i] -= learning_rate * (weight_gradients[i] + l2_penalty)
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data"""
        output, _, _ = self.forward(X)
        return output
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = 'mse') -> float:
        """Compute loss between predictions and targets"""
        if loss_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif loss_type == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif loss_type == 'binary_crossentropy':
            # Clip to prevent log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Compute accuracy for classification tasks"""
        if self.output_size == 1:
            # Binary classification
            predictions = (y_pred > threshold).astype(int)
            return np.mean(predictions.flatten() == y_true.flatten())
        else:
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            targets = np.argmax(y_true, axis=1)
            return np.mean(predictions == targets)
    
    def save_model(self, filepath: str):
        """Save the trained model to a file"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.training_history = model_data['training_history']
    
    def get_architecture_info(self) -> dict:
        """Get information about the network architecture"""
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
        return {
            'layer_sizes': self.layer_sizes,
            'total_parameters': total_params,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'depth': len(self.hidden_sizes),
            'width': max(self.hidden_sizes) if self.hidden_sizes else 0
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing FC-FFNN Implementation...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 1)
    
    # Create network
    network = FCFFNN(
        input_size=5,
        hidden_sizes=[10, 8],
        output_size=1,
        activation='relu',
        random_seed=42
    )
    
    # Test forward pass
    output, activations, z_values = network.forward(X)
    print(f"Forward pass output shape: {output.shape}")
    
    # Test backward pass
    weight_grads, bias_grads = network.backward(X, y, activations, z_values)
    print(f"Number of weight gradients: {len(weight_grads)}")
    
    # Test prediction
    predictions = network.predict(X[:5])
    print(f"Sample predictions: {predictions[:3].flatten()}")
    
    # Test loss computation
    loss = network.compute_loss(y, output)
    print(f"Sample loss: {loss:.4f}")
    
    # Print architecture info
    arch_info = network.get_architecture_info()
    print(f"Architecture: {arch_info}")
    
    print("FC-FFNN implementation test completed successfully!")