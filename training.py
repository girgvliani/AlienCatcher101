"""
Training module for FC-FFNN
Implements various training algorithms and utilities
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from ffnn import FCFFNN
import matplotlib.pyplot as plt
import time

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None
        self.best_biases = None
    
    def __call__(self, val_loss: float, model: FCFFNN) -> bool:
        """Check if training should stop early"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = [w.copy() for w in model.weights]
                self.best_biases = [b.copy() for b in model.biases]
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.weights = self.best_weights
                model.biases = self.best_biases
            return True
        return False

class LearningRateScheduler:
    """Learning rate scheduling strategies"""
    
    @staticmethod
    def step_decay(initial_lr: float, drop_rate: float = 0.5, epochs_drop: int = 10):
        """Step decay scheduler"""
        def scheduler(epoch: int) -> float:
            return initial_lr * (drop_rate ** (epoch // epochs_drop))
        return scheduler
    
    @staticmethod
    def exponential_decay(initial_lr: float, decay_rate: float = 0.95):
        """Exponential decay scheduler"""
        def scheduler(epoch: int) -> float:
            return initial_lr * (decay_rate ** epoch)
        return scheduler
    
    @staticmethod
    def cosine_annealing(initial_lr: float, T_max: int):
        """Cosine annealing scheduler"""
        def scheduler(epoch: int) -> float:
            return initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
        return scheduler

def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create mini-batches from data"""
    n_samples = X.shape[0]
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
    
    batches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batches.append((X[i:end_idx], y[i:end_idx]))
    
    return batches

def train_network(model: FCFFNN,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 l2_lambda: float = 0.0,
                 loss_type: str = 'mse',
                 early_stopping: Optional[EarlyStopping] = None,
                 lr_scheduler: Optional[Callable] = None,
                 verbose: bool = True,
                 plot_history: bool = True) -> Dict:
    """
    Train the neural network
    
    Args:
        model: FC-FFNN model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Initial learning rate
        l2_lambda: L2 regularization parameter
        loss_type: Type of loss function ('mse', 'mae', 'binary_crossentropy')
        early_stopping: Early stopping callback
        lr_scheduler: Learning rate scheduler function
        verbose: Whether to print training progress
        plot_history: Whether to plot training history
        
    Returns:
        Training history dictionary
    """
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    # Training loop
    start_time = time.time()
    current_lr = learning_rate
    
    for epoch in range(epochs):
        # Update learning rate if scheduler is provided
        if lr_scheduler is not None:
            current_lr = lr_scheduler(epoch)
        
        history['learning_rates'].append(current_lr)
        
        # Training phase
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        num_batches = 0
        
        # Create mini-batches
        batches = create_batches(X_train, y_train, batch_size)
        
        for batch_X, batch_y in batches:
            # Forward pass
            output, activations, z_values = model.forward(batch_X)
            
            # Compute loss
            batch_loss = model.compute_loss(batch_y, output, loss_type)
            epoch_train_loss += batch_loss
            
            # Compute accuracy (for classification tasks)
            if loss_type in ['binary_crossentropy']:
                batch_accuracy = model.compute_accuracy(batch_y, output)
                epoch_train_accuracy += batch_accuracy
            
            # Backward pass
            weight_grads, bias_grads = model.backward(batch_X, batch_y, activations, z_values)
            
            # Update parameters
            model.update_parameters(weight_grads, bias_grads, current_lr, l2_lambda)
            
            num_batches += 1
        
        # Average training metrics
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_accuracy = epoch_train_accuracy / num_batches if loss_type in ['binary_crossentropy'] else 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_accuracy)
        
        # Validation phase
        val_loss = 0.0
        val_accuracy = 0.0
        
        if X_val is not None and y_val is not None:
            val_output = model.predict(X_val)
            val_loss = model.compute_loss(y_val, val_output, loss_type)
            
            if loss_type in ['binary_crossentropy']:
                val_accuracy = model.compute_accuracy(y_val, val_output)
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"LR: {current_lr:.6f}")
            
            if loss_type in ['binary_crossentropy']:
                print(f"Train Acc: {avg_train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")
        
        # Early stopping check
        if early_stopping is not None and X_val is not None:
            if early_stopping(val_loss, model):
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Training completed
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if X_val is not None:
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    # Update model's training history
    model.training_history = history
    
    # Plot training history
    if plot_history:
        plot_training_history(history, loss_type)
    
    return history

def plot_training_history(history: Dict, loss_type: str):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
    if history['val_loss'][0] != 0:  # If validation data was provided
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy (if applicable)
    if loss_type in ['binary_crossentropy'] and max(history['train_accuracy']) > 0:
        axes[1].plot(history['train_accuracy'], label='Training Accuracy', color='blue')
        if history['val_accuracy'][0] != 0:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Plot learning rate instead
        axes[1].plot(history['learning_rates'], color='green')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def cross_validation(model_func: Callable,
                    X: np.ndarray,
                    y: np.ndarray,
                    k_folds: int = 5,
                    **train_kwargs) -> Dict:
    """
    Perform k-fold cross-validation
    
    Args:
        model_func: Function that returns a new model instance
        X: Feature data
        y: Target data
        k_folds: Number of folds
        **train_kwargs: Arguments to pass to train_network
        
    Returns:
        Cross-validation results
    """
    
    n_samples = X.shape[0]
    fold_size = n_samples // k_folds
    
    cv_scores = []
    fold_histories = []
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for fold in range(k_folds):
        print(f"Training fold {fold + 1}/{k_folds}...")
        
        # Create train/validation split
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        
        X_val_fold = X_shuffled[start_idx:end_idx]
        y_val_fold = y_shuffled[start_idx:end_idx]
        
        X_train_fold = np.concatenate([
            X_shuffled[:start_idx],
            X_shuffled[end_idx:]
        ], axis=0)
        y_train_fold = np.concatenate([
            y_shuffled[:start_idx],
            y_shuffled[end_idx:]
        ], axis=0)
        
        # Create and train model
        model = model_func()
        
        # Train model (suppress verbose output for CV)
        history = train_network(
            model, X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            verbose=False, plot_history=False,
            **train_kwargs
        )
        
        # Store results
        val_score = history['val_loss'][-1]
        cv_scores.append(val_score)
        fold_histories.append(history)
    
    cv_results = {
        'scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'fold_histories': fold_histories
    }
    
    print(f"Cross-validation completed!")
    print(f"Mean CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    return cv_results

def grid_search(model_func: Callable,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               param_grid: Dict,
               scoring: str = 'loss') -> Dict:
    """
    Perform grid search for hyperparameter tuning
    
    Args:
        model_func: Function that creates model with given parameters
        X_train, y_train: Training data
        X_val, y_val: Validation data
        param_grid: Dictionary of parameters to search
        scoring: Scoring metric ('loss' or 'accuracy')
        
    Returns:
        Grid search results
    """
    
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = np.inf if scoring == 'loss' else -np.inf
    best_params = None
    best_model = None
    all_results = []
    
    print(f"Starting grid search with {len(param_combinations)} combinations...")
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Create model with current parameters
            model = model_func(**params)
            
            # Train model
            history = train_network(
                model, X_train, y_train, X_val, y_val,
                verbose=False, plot_history=False,
                epochs=50  # Reduce epochs for grid search
            )
            
            # Get score
            if scoring == 'loss':
                score = history['val_loss'][-1]
                is_better = score < best_score
            else:  # accuracy
                score = history['val_accuracy'][-1]
                is_better = score > best_score
            
            # Store results
            result = {
                'params': params,
                'score': score,
                'history': history
            }
            all_results.append(result)
            
            # Update best model
            if is_better:
                best_score = score
                best_params = params
                best_model = model
                
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue
    
    grid_results = {
        'best_params': best_params,
        'best_score': best_score,
        'best_model': best_model,
        'all_results': all_results
    }
    
    print(f"Grid search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return grid_results

if __name__ == "__main__":
    # Example usage
    from ffnn import FCFFNN
    
    print("Testing training module...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = (X[:, 0] + X[:, 1]**2 - X[:, 2] + np.random.randn(1000) * 0.1).reshape(-1, 1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create model
    model = FCFFNN(
        input_size=5,
        hidden_sizes=[20, 15],
        output_size=1,
        activation='relu'
    )
    
    # Set up early stopping and learning rate scheduler
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    lr_scheduler = LearningRateScheduler.exponential_decay(0.01, 0.95)
    
    # Train model
    history = train_network(
        model, X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        verbose=True
    )
    
    print("Training module test completed successfully!")