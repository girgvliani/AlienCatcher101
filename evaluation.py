"""
Model evaluation and metrics module for FC-FFNN
Provides comprehensive evaluation tools and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from ffnn import FCFFNN
import time

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model: FCFFNN):
        self.model = model
        self.evaluation_results = {}
    
    def evaluate_regression(self, 
                          X_test: np.ndarray, 
                          y_test: np.ndarray,
                          verbose: bool = True) -> Dict:
        """
        Comprehensive evaluation for regression tasks
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Whether to print results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # Explained Variance Score
        var_y = np.var(y_test)
        var_residual = np.var(y_test - y_pred)
        explained_var = 1 - (var_residual / (var_y + 1e-8))
        
        # Maximum error
        max_error = np.max(np.abs(y_test - y_pred))
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'explained_variance': explained_var,
            'max_error': max_error,
            'prediction_time': prediction_time,
            'samples_per_second': len(X_test) / prediction_time
        }
        
        if verbose:
            print("=== Regression Evaluation Results ===")
            print(f"Mean Squared Error (MSE): {mse:.6f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
            print(f"Mean Absolute Error (MAE): {mae:.6f}")
            print(f"R² Score: {r2:.6f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Explained Variance Score: {explained_var:.6f}")
            print(f"Maximum Error: {max_error:.6f}")
            print(f"Prediction Time: {prediction_time:.4f} seconds")
            print(f"Throughput: {results['samples_per_second']:.0f} samples/second")
        
        self.evaluation_results['regression'] = results
        return results
    
    def evaluate_classification(self, 
                              X_test: np.ndarray, 
                              y_test: np.ndarray,
                              threshold: float = 0.5,
                              verbose: bool = True) -> Dict:
        """
        Comprehensive evaluation for classification tasks
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Classification threshold
            verbose: Whether to print results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        # Make predictions
        start_time = time.time()
        y_pred_proba = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Convert to binary predictions
        if self.model.output_size == 1:
            y_pred = (y_pred_proba > threshold).astype(int)
            y_true = y_test.astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
        
        # Calculate confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        accuracy = np.mean(y_pred.flatten() == y_true.flatten())
        
        if self.model.output_size == 1:  # Binary classification
            tn, fp, fn, tp = confusion_matrix.ravel()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            
            # AUC-ROC calculation
            auc_roc = self._calculate_auc_roc(y_true, y_pred_proba)
            
        else:  # Multi-class classification
            precision, recall, f1 = self._calculate_multiclass_metrics(confusion_matrix)
            specificity = None
            auc_roc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc_roc,
            'confusion_matrix': confusion_matrix,
            'prediction_time': prediction_time,
            'samples_per_second': len(X_test) / prediction_time
        }
        
        if verbose:
            print("=== Classification Evaluation Results ===")
            print(f"Accuracy: {accuracy:.6f}")
            print(f"Precision: {precision:.6f}")
            print(f"Recall: {recall:.6f}")
            print(f"F1 Score: {f1:.6f}")
            if specificity is not None:
                print(f"Specificity: {specificity:.6f}")
            if auc_roc is not None:
                print(f"AUC-ROC: {auc_roc:.6f}")
            print(f"Prediction Time: {prediction_time:.4f} seconds")
            print(f"Throughput: {results['samples_per_second']:.0f} samples/second")
        
        self.evaluation_results['classification'] = results
        return results
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
        n_classes = len(classes)
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true_flat == true_class) & (y_pred_flat == pred_class))
        
        return cm
    
    def _calculate_multiclass_metrics(self, confusion_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 for multiclass classification"""
        n_classes = confusion_matrix.shape[0]
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Return macro averages
        return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)
    
    def _calculate_auc_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate AUC-ROC score for binary classification"""
        # Simple implementation of AUC-ROC
        y_true_flat = y_true.flatten()
        y_scores_flat = y_scores.flatten()
        
        # Sort by predicted probabilities
        sorted_indices = np.argsort(y_scores_flat)
        y_true_sorted = y_true_flat[sorted_indices]
        
        # Calculate TPR and FPR for different thresholds
        n_pos = np.sum(y_true_flat == 1)
        n_neg = np.sum(y_true_flat == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5  # Random performance when only one class present
        
        tpr_values = []
        fpr_values = []
        
        for i in range(len(y_true_sorted) + 1):
            if i == 0:
                tp = fn = fp = tn = 0
            else:
                # Predictions above current threshold are positive
                predictions = np.zeros_like(y_true_sorted)
                predictions[i:] = 1
                
                tp = np.sum((y_true_sorted == 1) & (predictions == 1))
                fn = np.sum((y_true_sorted == 1) & (predictions == 0))
                fp = np.sum((y_true_sorted == 0) & (predictions == 1))
                tn = np.sum((y_true_sorted == 0) & (predictions == 0))
            
            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return abs(auc)
    
    def plot_predictions(self, 
                        X_test: np.ndarray, 
                        y_test: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        max_samples: int = 1000):
        """Plot prediction results"""
        
        y_pred = self.model.predict(X_test)
        
        # Limit samples for plotting
        if len(X_test) > max_samples:
            indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_plot = X_test[indices]
            y_test_plot = y_test[indices]
            y_pred_plot = y_pred[indices]
        else:
            X_plot = X_test
            y_test_plot = y_test
            y_pred_plot = y_pred
        
        if self.model.output_size == 1:  # Regression or binary classification
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Prediction vs True values
            axes[0, 0].scatter(y_test_plot, y_pred_plot, alpha=0.6)
            axes[0, 0].plot([y_test_plot.min(), y_test_plot.max()], 
                           [y_test_plot.min(), y_test_plot.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('True Values')
            axes[0, 0].set_ylabel('Predictions')
            axes[0, 0].set_title('Prediction vs True Values')
            axes[0, 0].grid(True)
            
            # Residuals plot
            residuals = y_test_plot - y_pred_plot
            axes[0, 1].scatter(y_pred_plot, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predictions')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            axes[0, 1].grid(True)
            
            # Residuals histogram
            axes[1, 0].hist(residuals.flatten(), bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].grid(True)
            
            # Feature importance (correlation with residuals)
            if X_plot.shape[1] <= 10:  # Only if reasonable number of features
                feature_correlations = []
                for i in range(X_plot.shape[1]):
                    corr = np.corrcoef(X_plot[:, i], residuals.flatten())[0, 1]
                    feature_correlations.append(abs(corr))
                
                feature_names_plot = feature_names if feature_names else [f'Feature {i}' for i in range(X_plot.shape[1])]
                
                axes[1, 1].bar(range(len(feature_correlations)), feature_correlations)
                axes[1, 1].set_xlabel('Features')
                axes[1, 1].set_ylabel('|Correlation with Residuals|')
                axes[1, 1].set_title('Feature Correlation with Residuals')
                axes[1, 1].set_xticks(range(len(feature_names_plot)))
                axes[1, 1].set_xticklabels(feature_names_plot, rotation=45)
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'Too many features\nto display', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Feature Analysis')
        
        else:  # Multi-class classification
            # Confusion matrix heatmap
            if hasattr(self, 'evaluation_results') and 'classification' in self.evaluation_results:
                cm = self.evaluation_results['classification']['confusion_matrix']
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=[f'Class {i}' for i in range(cm.shape[1])],
                           yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_errors(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      top_k: int = 10) -> Dict:
        """Analyze prediction errors in detail"""
        
        y_pred = self.model.predict(X_test)
        
        if self.model.output_size == 1:  # Regression
            errors = np.abs(y_test - y_pred).flatten()
            
            # Find worst predictions
            worst_indices = np.argsort(errors)[-top_k:]
            
            error_analysis = {
                'worst_predictions': {
                    'indices': worst_indices,
                    'errors': errors[worst_indices],
                    'true_values': y_test[worst_indices].flatten(),
                    'predictions': y_pred[worst_indices].flatten(),
                    'inputs': X_test[worst_indices]
                },
                'error_statistics': {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors),
                    'percentiles': {
                        '50th': np.percentile(errors, 50),
                        '90th': np.percentile(errors, 90),
                        '95th': np.percentile(errors, 95),
                        '99th': np.percentile(errors, 99)
                    }
                }
            }
            
            print("=== Error Analysis ===")
            print(f"Mean absolute error: {error_analysis['error_statistics']['mean_error']:.6f}")
            print(f"Error std deviation: {error_analysis['error_statistics']['std_error']:.6f}")
            print(f"90th percentile error: {error_analysis['error_statistics']['percentiles']['90th']:.6f}")
            print(f"95th percentile error: {error_analysis['error_statistics']['percentiles']['95th']:.6f}")
            print(f"Worst {top_k} prediction errors: {error_analysis['worst_predictions']['errors']}")
            
        else:  # Classification
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Find misclassified samples
            misclassified = y_pred_classes != y_true_classes
            misclassified_indices = np.where(misclassified)[0]
            
            if len(misclassified_indices) > top_k:
                # Get confidence scores for misclassified samples
                misclass_confidences = np.max(y_pred[misclassified_indices], axis=1)
                # Sort by confidence (most confident wrong predictions first)
                conf_sorted_indices = np.argsort(misclass_confidences)[-top_k:]
                worst_indices = misclassified_indices[conf_sorted_indices]
            else:
                worst_indices = misclassified_indices
            
            error_analysis = {
                'misclassified_samples': {
                    'indices': worst_indices,
                    'true_classes': y_true_classes[worst_indices],
                    'predicted_classes': y_pred_classes[worst_indices],
                    'prediction_confidences': np.max(y_pred[worst_indices], axis=1),
                    'inputs': X_test[worst_indices]
                },
                'error_statistics': {
                    'total_misclassified': len(misclassified_indices),
                    'misclassification_rate': len(misclassified_indices) / len(y_test),
                    'avg_confidence_correct': np.mean(np.max(y_pred[~misclassified], axis=1)),
                    'avg_confidence_wrong': np.mean(np.max(y_pred[misclassified], axis=1)) if np.any(misclassified) else 0
                }
            }
            
            print("=== Error Analysis ===")
            print(f"Total misclassified: {error_analysis['error_statistics']['total_misclassified']}")
            print(f"Misclassification rate: {error_analysis['error_statistics']['misclassification_rate']:.4f}")
            print(f"Avg confidence (correct): {error_analysis['error_statistics']['avg_confidence_correct']:.4f}")
            print(f"Avg confidence (wrong): {error_analysis['error_statistics']['avg_confidence_wrong']:.4f}")
        
        return error_analysis
    
    def generate_report(self, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       task_type: str = 'regression',
                       save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 60)
        report.append("NEURAL NETWORK EVALUATION REPORT")
        report.append("=" * 60)
        
        # Model architecture info
        arch_info = self.model.get_architecture_info()
        report.append("\n--- MODEL ARCHITECTURE ---")
        report.append(f"Layer sizes: {arch_info['layer_sizes']}")
        report.append(f"Total parameters: {arch_info['total_parameters']:,}")
        report.append(f"Activation function: {arch_info['activation']}")
        report.append(f"Output activation: {arch_info['output_activation']}")
        report.append(f"Network depth: {arch_info['depth']}")
        report.append(f"Network width: {arch_info['width']}")
        
        # Evaluation metrics
        if task_type == 'regression':
            results = self.evaluate_regression(X_test, y_test, verbose=False)
            report.append("\n--- REGRESSION METRICS ---")
            report.append(f"R² Score: {results['r2_score']:.6f}")
            report.append(f"RMSE: {results['rmse']:.6f}")
            report.append(f"MAE: {results['mae']:.6f}")
            report.append(f"MAPE: {results['mape']:.2f}%")
        else:
            results = self.evaluate_classification(X_test, y_test, verbose=False)
            report.append("\n--- CLASSIFICATION METRICS ---")
            report.append(f"Accuracy: {results['accuracy']:.6f}")
            report.append(f"Precision: {results['precision']:.6f}")
            report.append(f"Recall: {results['recall']:.6f}")
            report.append(f"F1 Score: {results['f1_score']:.6f}")
        
        # Performance metrics
        report.append("\n--- PERFORMANCE METRICS ---")
        report.append(f"Prediction time: {results['prediction_time']:.4f} seconds")
        report.append(f"Throughput: {results['samples_per_second']:.0f} samples/second")
        
        # Error analysis
        error_analysis = self.analyze_errors(X_test, y_test, verbose=False)
        report.append("\n--- ERROR ANALYSIS ---")
        if task_type == 'regression':
            stats = error_analysis['error_statistics']
            report.append(f"Mean absolute error: {stats['mean_error']:.6f}")
            report.append(f"Error std deviation: {stats['std_error']:.6f}")
            report.append(f"95th percentile error: {stats['percentiles']['95th']:.6f}")
        else:
            stats = error_analysis['error_statistics']
            report.append(f"Misclassification rate: {stats['misclassification_rate']:.4f}")
            report.append(f"Confidence gap: {stats['avg_confidence_correct'] - stats['avg_confidence_wrong']:.4f}")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text

def compare_models(models: List[FCFFNN], 
                  model_names: List[str],
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  task_type: str = 'regression') -> Dict:
    """Compare multiple models on the same test set"""
    
    comparison_results = {}
    
    for model, name in zip(models, model_names):
        evaluator = ModelEvaluator(model)
        
        if task_type == 'regression':
            results = evaluator.evaluate_regression(X_test, y_test, verbose=False)
            key_metric = results['r2_score']
        else:
            results = evaluator.evaluate_classification(X_test, y_test, verbose=False)
            key_metric = results['accuracy']
        
        comparison_results[name] = {
            'results': results,
            'key_metric': key_metric,
            'architecture': model.get_architecture_info()
        }
    
    # Sort by key metric
    sorted_models = sorted(comparison_results.items(), 
                          key=lambda x: x[1]['key_metric'], 
                          reverse=True)
    
    print("=== MODEL COMPARISON ===")
    metric_name = 'R² Score' if task_type == 'regression' else 'Accuracy'
    
    for i, (name, data) in enumerate(sorted_models):
        print(f"{i+1}. {name}: {metric_name} = {data['key_metric']:.6f}")
        print(f"   Parameters: {data['architecture']['total_parameters']:,}")
        if task_type == 'regression':
            print(f"   RMSE: {data['results']['rmse']:.6f}")
        else:
            print(f"   F1 Score: {data['results']['f1_score']:.6f}")
        print()
    
    return comparison_results

if __name__ == "__main__":
    # Example usage
    from ffnn import FCFFNN
    
    print("Testing evaluation module...")
    
    # Generate sample regression data
    np.random.seed(42)
    X_test = np.random.randn(200, 5)
    y_test = (X_test[:, 0] + X_test[:, 1]**2 - X_test[:, 2] + np.random.randn(200) * 0.1).reshape(-1, 1)
    
    # Create and evaluate model
    model = FCFFNN(input_size=5, hidden_sizes=[20, 15], output_size=1)
    evaluator = ModelEvaluator(model)
    
    # Evaluate regression
    regression_results = evaluator.evaluate_regression(X_test, y_test)
    
    # Generate report
    report = evaluator.generate_report(X_test, y_test, task_type='regression')
    print("\n" + report)
    
    print("Evaluation module test completed successfully!")