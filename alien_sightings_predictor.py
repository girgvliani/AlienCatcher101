"""
Alien Sightings Prediction Application
Real-world application of FC-FFNN for predicting UFO sighting patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings
import pickle
warnings.filterwarnings('ignore')

# Try to import pandas, create simple alternative if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using simplified DataFrame alternative")

# Import local modules with error handling
try:
    from ffnn import FCFFNN
    from training import train_network, EarlyStopping, LearningRateScheduler
    from evaluation import ModelEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are in the same directory or Python path")
    raise

# Simple DataFrame alternative if pandas is not available
class SimpleDataFrame:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = data
        else:
            self.data = [data] if not isinstance(data, list) else data
    
    def head(self, n=5):
        return SimpleDataFrame(self.data[:n])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # Column access
            return [row.get(key) for row in self.data if isinstance(row, dict)]
        else:
            # Row access
            return SimpleDataFrame(self.data[key])
    
    def groupby(self, column):
        groups = {}
        for row in self.data:
            if isinstance(row, dict):
                key = row.get(column)
                if key not in groups:
                    groups[key] = []
                groups[key].append(row)
        return groups
    
    def __str__(self):
        return str(self.data[:5]) + "..." if len(self.data) > 5 else str(self.data)

class AlienSightingsPredictor:
    """
    Predicts alien/UFO sighting likelihood based on various features
    """
    
    def _calculate_sighting_probability(self, days, hours, lunar_phase, season, 
                                      is_weekend, population_density, military_proximity, 
                                      weather_clear):
        """Calculate base sighting probability based on features"""
        
        # Base probability
        base_prob = 0.1
        
        # Time patterns (higher at night, especially late night)
        hour_effect = np.where(
            (hours >= 22) | (hours <= 4), 0.3,  # Night time boost
            np.where((hours >= 18) | (hours <= 6), 0.2, 0.1)  # Evening/early morning
        )
        
        # Lunar phase effect (higher during full moon)
        lunar_effect = 0.1 + 0.2 * np.sin(2 * np.pi * lunar_phase)
        
        # Seasonal effects (higher in summer/fall)
        seasonal_multiplier = np.where(
            season == 2, 1.3,  # Summer
            np.where(season == 3, 1.1, 0.9)  # Fall, otherwise lower
        )
        
        # Weekend effect (slightly higher on weekends)
        weekend_effect = np.where(is_weekend, 1.1, 1.0)
        
        # Population density (moderate density optimal)
        pop_effect = np.exp(-((np.log(population_density + 1) - 3) ** 2) / 8)
        
        # Military proximity (higher near military bases)
        military_effect = 1 + 0.3 * np.exp(-military_proximity / 30)
        
        # Weather effect (higher probability in clear weather)
        weather_effect = np.where(weather_clear, 1.2, 0.8)
        
        # Combine all effects
        probability = (base_prob + hour_effect + lunar_effect) * \
                     seasonal_multiplier * weekend_effect * pop_effect * \
                     military_effect * weather_effect
        
        return np.clip(probability, 0, 1)
    
    def preprocess_features(self, X: np.ndarray, fit_transform: bool = False) -> np.ndarray:
        """Normalize features for better training"""
        
        if fit_transform:
            # Simple standardization
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8
        
        # Apply standardization
        X_scaled = (X - self.feature_means) / self.feature_stds
        
        return X_scaled
    
    def create_model(self, input_size: int, architecture: str = 'medium') -> FCFFNN:
        """Create neural network model with specified architecture"""
        
        architectures = {
            'small': [16, 8],
            'medium': [32, 16, 8],
            'large': [64, 32, 16],
            'deep': [32, 24, 16, 12, 8]
        }
        
        hidden_sizes = architectures.get(architecture, architectures['medium'])
        
        model = FCFFNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation='relu',
            output_activation='sigmoid',  # For binary classification
            random_seed=42
        )
        
        return model
    
    def train_predictor(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       architecture: str = 'medium',
                       epochs: int = 150,
                       learning_rate: float = 0.001) -> Dict:
        """Train the alien sightings predictor"""
        
        print("Training Alien Sightings Predictor...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Architecture: {architecture}")
        
        # Preprocess features
        X_train_scaled = self.preprocess_features(X_train, fit_transform=True)
        X_val_scaled = self.preprocess_features(X_val, fit_transform=False)
        
        # Create model
        self.model = self.create_model(X_train.shape[1], architecture)
        
        # Set up training
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)
        lr_scheduler = LearningRateScheduler.exponential_decay(learning_rate, 0.95)
        
        # Train model
        history = train_network(
            self.model, X_train_scaled, y_train, X_val_scaled, y_val,
            epochs=epochs,
            batch_size=64,
            learning_rate=learning_rate,
            l2_lambda=0.001,
            loss_type='binary_crossentropy',
            early_stopping=early_stopping,
            lr_scheduler=lr_scheduler,
            verbose=True
        )
        
        return history
    
    def evaluate_predictor(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the trained predictor"""
        
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Preprocess test features
        X_test_scaled = self.preprocess_features(X_test, fit_transform=False)
        
        # Evaluate model
        evaluator = ModelEvaluator(self.model)
        results = evaluator.evaluate_classification(X_test_scaled, y_test)
        
        # Additional analysis
        y_pred_proba = self.model.predict(X_test_scaled)
        
        # Feature importance analysis
        feature_importance = self.analyze_feature_importance(X_test_scaled, y_test)
        results['feature_importance'] = feature_importance
        
        return results
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Analyze feature importance using perturbation method"""
        
        baseline_pred = self.model.predict(X)
        baseline_accuracy = np.mean((baseline_pred > 0.5) == y.flatten())
        
        feature_importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Perturb feature by adding noise
            X_perturbed = X.copy()
            X_perturbed[:, i] += np.random.normal(0, np.std(X[:, i]), len(X))
            
            # Get predictions with perturbed feature
            perturbed_pred = self.model.predict(X_perturbed)
            perturbed_accuracy = np.mean((perturbed_pred > 0.5) == y.flatten())
            
            # Importance is the drop in accuracy
            importance = baseline_accuracy - perturbed_accuracy
            feature_importance[feature_name] = importance
        
        return feature_importance
    
    def predict_sighting_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict sighting probability for given features"""
        
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess features
        features_scaled = self.preprocess_features(features, fit_transform=False)
        
        # Get probability predictions
        probabilities = self.model.predict(features_scaled)
        
        return probabilities
    
    def generate_sighting_forecast(self, start_date: str, days: int = 30):
        """Generate sighting probability forecast for upcoming days"""
        
        forecasts = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        for day_offset in range(days):
            current_date = start_dt + timedelta(days=day_offset)
            
            # Generate features for each hour of the day
            for hour in range(24):
                day_of_year = current_date.timetuple().tm_yday
                day_of_week = current_date.weekday()
                
                # Simplified lunar phase calculation
                lunar_phase = ((day_of_year % 29.5) / 29.5)
                
                season = ((day_of_year - 1) // 91) % 4
                is_weekend = 1 if day_of_week >= 5 else 0
                
                # Use average values for other features
                features = np.array([[
                    day_of_year, hour, day_of_week, lunar_phase, season, is_weekend,
                    100,  # population_density
                    25,   # military_proximity
                    1,    # weather_clear
                    15,   # temperature
                    0.6   # humidity
                ]])
                
                probability = self.predict_sighting_probability(features)[0, 0]
                
                forecasts.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'datetime': current_date.strftime('%Y-%m-%d %H:00'),
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'sighting_probability': probability,
                    'risk_level': self._categorize_risk(probability)
                })
        
        # Return pandas DataFrame if available, otherwise simple alternative
        if HAS_PANDAS:
            return pd.DataFrame(forecasts)
        else:
            return SimpleDataFrame(forecasts)
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize sighting probability into risk levels"""
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.5:
            return 'Medium'
        elif probability >= 0.3:
            return 'Low'
        else:
            return 'Very Low'
    
    def plot_feature_importance(self, feature_importance: Dict):
        """Plot feature importance analysis"""
        
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance_values)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(features)), [importance_values[i] for i in sorted_indices])
        plt.xlabel('Features')
        plt.ylabel('Importance (Accuracy Drop)')
        plt.title('Feature Importance for Alien Sighting Prediction')
        plt.xticks(range(len(features)), [features[i] for i in sorted_indices], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_sighting_forecast(self, forecast_df):
        """Plot sighting probability forecast"""
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            if HAS_PANDAS and hasattr(forecast_df, 'groupby'):
                # Pandas DataFrame - full functionality
                daily_max = forecast_df.groupby('date')['sighting_probability'].max()
                axes[0].plot(daily_max.index, daily_max.values, marker='o', linewidth=2)
                axes[0].set_title('Daily Maximum Sighting Probability')
                axes[0].set_ylabel('Probability')
                axes[0].grid(True, alpha=0.3)
                axes[0].tick_params(axis='x', rotation=45)
                
                # Hourly heatmap for first week
                first_week = forecast_df[forecast_df['date'].isin(daily_max.index[:7])]
                if len(first_week) > 0:
                    heatmap_data = first_week.pivot(index='hour', columns='date', values='sighting_probability')
                    
                    im = axes[1].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
                    axes[1].set_title('Hourly Sighting Probability Heatmap (First Week)')
                    axes[1].set_xlabel('Date')
                    axes[1].set_ylabel('Hour of Day')
                    axes[1].set_xticks(range(len(heatmap_data.columns)))
                    axes[1].set_xticklabels(heatmap_data.columns, rotation=45)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=axes[1], label='Sighting Probability')
                else:
                    axes[1].text(0.5, 0.5, 'Insufficient data for heatmap', 
                               ha='center', va='center', transform=axes[1].transAxes)
            else:
                # Simple alternative for non-pandas data
                dates = []
                probabilities = []
                
                if hasattr(forecast_df, 'data'):
                    data = forecast_df.data
                else:
                    data = forecast_df
                
                # Extract daily maximum probabilities
                daily_data = {}
                for row in data:
                    if isinstance(row, dict):
                        date = row.get('date')
                        prob = row.get('sighting_probability', 0)
                        if date not in daily_data or prob > daily_data[date]:
                            daily_data[date] = prob
                
                sorted_dates = sorted(daily_data.keys())
                dates = sorted_dates
                probabilities = [daily_data[date] for date in sorted_dates]
                
                axes[0].plot(range(len(dates)), probabilities, marker='o', linewidth=2)
                axes[0].set_title('Daily Maximum Sighting Probability')
                axes[0].set_ylabel('Probability')
                axes[0].set_xlabel('Days')
                axes[0].grid(True, alpha=0.3)
                
                # Simple hourly average plot
                hourly_avg = [0] * 24
                hourly_count = [0] * 24
                
                for row in data:
                    if isinstance(row, dict):
                        hour = row.get('hour')
                        prob = row.get('sighting_probability', 0)
                        if hour is not None and 0 <= hour < 24:
                            hourly_avg[hour] += prob
                            hourly_count[hour] += 1
                
                # Calculate averages
                for i in range(24):
                    if hourly_count[i] > 0:
                        hourly_avg[i] /= hourly_count[i]
                
                axes[1].bar(range(24), hourly_avg)
                axes[1].set_title('Average Hourly Sighting Probability')
                axes[1].set_xlabel('Hour of Day')
                axes[1].set_ylabel('Average Probability')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create forecast plot: {e}")
            print("Forecast data summary:")
            if hasattr(forecast_df, 'data'):
                print(f"Total records: {len(forecast_df.data)}")
                if forecast_df.data:
                    print(f"Sample record: {forecast_df.data[0]}")
            else:
                print(f"Forecast object type: {type(forecast_df)}")
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessing parameters"""
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            # Save model
            self.model.save_model(filepath)
            
            # Save preprocessing parameters
            preprocessing_data = {
                'feature_means': getattr(self, 'feature_means', None),
                'feature_stds': getattr(self, 'feature_stds', None),
                'feature_names': self.feature_names
            }
            
            preprocessing_filepath = filepath.replace('.pkl', '_preprocessing.pkl')
            with open(preprocessing_filepath, 'wb') as f:
                pickle.dump(preprocessing_data, f)
            
            print(f"Model and preprocessing data saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessing parameters"""
        try:
            # Load model
            self.model = FCFFNN(input_size=11, hidden_sizes=[32, 16, 8], output_size=1)
            self.model.load_model(filepath)
            
            # Load preprocessing parameters
            preprocessing_filepath = filepath.replace('.pkl', '_preprocessing.pkl')
            with open(preprocessing_filepath, 'rb') as f:
                preprocessing_data = pickle.load(f)
            
            self.feature_means = preprocessing_data.get('feature_means')
            self.feature_stds = preprocessing_data.get('feature_stds')
            self.feature_names = preprocessing_data.get('feature_names', self.feature_names)
            
            print(f"Model and preprocessing data loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

def main():
    """Main function to demonstrate the alien sightings predictor"""
    
    print("=== Alien Sightings Predictor Demo ===\n")
    
    try:
        # Initialize predictor
        predictor = AlienSightingsPredictor()
        
        # Generate synthetic data
        print("1. Generating synthetic alien sighting data...")
        X, y = predictor.generate_synthetic_data(n_samples=8000)
        
        print(f"Generated {len(X)} samples with {X.shape[1]} features")
        print(f"Positive sighting rate: {np.mean(y):.3f}")
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        
        # Train predictor
        print("2. Training the neural network predictor...")
        history = predictor.train_predictor(
            X_train, y_train, X_val, y_val,
            architecture='medium',
            epochs=100,
            learning_rate=0.002
        )
        
        # Evaluate predictor
        print("\n3. Evaluating the trained predictor...")
        results = predictor.evaluate_predictor(X_test, y_test)
        
        # Feature importance analysis
        print("\n4. Analyzing feature importance...")
        try:
            predictor.plot_feature_importance(results['feature_importance'])
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
            print("Feature importance values:")
            for feature, importance in results['feature_importance'].items():
                print(f"  {feature}: {importance:.4f}")
        
        # Generate forecast
        print("\n5. Generating sighting forecast...")
        forecast = predictor.generate_sighting_forecast('2025-07-01', days=14)
        
        print("\nForecast sample:")
        if HAS_PANDAS:
            print(forecast.head(10))
        else:
            print("First 10 forecast entries:")
            for i, entry in enumerate(forecast.data[:10]):
                print(f"  {i+1}. {entry}")
        
        # Plot forecast
        try:
            predictor.plot_sighting_forecast(forecast)
        except Exception as e:
            print(f"Could not plot forecast: {e}")
        
        # High-risk periods
        if HAS_PANDAS:
            high_risk_periods = forecast[forecast['risk_level'] == 'High']
        else:
            high_risk_periods = [entry for entry in forecast.data if entry.get('risk_level') == 'High']
        
        print(f"\nHigh-risk periods in forecast: {len(high_risk_periods)}")
        if len(high_risk_periods) > 0:
            print("Top 5 high-risk periods:")
            if HAS_PANDAS:
                print(high_risk_periods[['datetime', 'sighting_probability']].head())
            else:
                for i, period in enumerate(high_risk_periods[:5]):
                    print(f"  {period['datetime']}: {period['sighting_probability']:.3f}")
        
        # Save model
        print("\n6. Saving trained model...")
        try:
            predictor.save_model('alien_sightings_model.pkl')
        except Exception as e:
            print(f"Could not save model: {e}")
        
        print("\n=== Demo completed successfully! ===")
        
        return predictor, results, forecast
        
    except Exception as e:
        print(f"Error in main demo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    predictor, results, forecast = main()

class AlienSightingsPredictor:
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = [
            'day_of_year', 'hour_of_day', 'day_of_week',
            'lunar_phase', 'season', 'is_weekend',
            'population_density', 'military_proximity',
            'weather_clear', 'temperature', 'humidity'
        ]
        
    def generate_synthetic_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic alien sighting data based on realistic patterns
        
        Args:
            n_samples: Number of data points to generate
            
        Returns:
            Features and targets for training
        """
        np.random.seed(42)
        
        # Generate temporal features
        days = np.random.randint(1, 366, n_samples)  # Day of year
        hours = np.random.randint(0, 24, n_samples)  # Hour of day
        day_of_week = np.random.randint(0, 7, n_samples)  # Day of week
        
        # Lunar phase (0-1, where 0 = New Moon, 0.5 = Full Moon)
        lunar_phase = np.random.uniform(0, 1, n_samples)
        
        # Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        season = ((days - 1) // 91) % 4
        
        # Weekend indicator
        is_weekend = (day_of_week >= 5).astype(int)
        
        # Geographic/demographic features
        population_density = np.random.lognormal(2, 1, n_samples)  # People per sq km
        military_proximity = np.random.exponential(50, n_samples)  # Distance to military base (km)
        
        # Weather features
        weather_clear = np.random.binomial(1, 0.7, n_samples)  # Clear weather probability
        temperature = np.random.normal(15, 10, n_samples)  # Temperature in Celsius
        humidity = np.random.uniform(0.3, 0.9, n_samples)  # Relative humidity
        
        # Combine features
        features = np.column_stack([
            days, hours, day_of_week, lunar_phase, season, is_weekend,
            population_density, military_proximity, weather_clear, 
            temperature, humidity
        ])
        
        # Generate sighting likelihood based on realistic patterns
        sighting_probability = self._calculate_sighting_probability(
            days, hours, lunar_phase, season, is_weekend,
            population_density, military_proximity, weather_clear
        )
        
        # Add some noise and convert to binary outcomes
        noise = np.random.normal(0, 0.1, n_samples)
        sighting_probability = np.clip(sighting_probability + noise, 0, 1)
        
        # Create binary targets (1 = sighting, 0 = no sighting)
        targets = np.random.binomial(1, sighting_probability, n_samples).reshape(-1, 1)
        
        return features.astype(np.float32), targets.astype(np.float32)
    
    def _calculate_sighting_probability(self, days, hours, lunar_phase, season, 
                                      is_weekend, population_density, military_proximity, 
                                      weather_clear):
        """Calculate base sighting probability based on features"""
        
        # Base probability
        base_prob = 0.1
        
        # Time patterns (higher at night, especially late night)
        hour_effect = np.where(
            (hours >= 22) | (hours <= 4), 0.3,  # Night time boost
            np.where((hours >= 18) | (hours <= 6), 0.2, 0.1)  # Evening/early morning
        )
        
        # Lunar phase effect (higher during full moon)
        lunar_effect = 0.1 + 0.2 * np.sin(2 * np.pi * lunar_phase)
        
        # Seasonal effects (higher in summer/fall)
        seasonal_multiplier = np.where(
            season == 2, 1.3,  # Summer
            np.where(season == 3, 1.1, 0.9)  # Fall, otherwise lower
        )
        
        # Weekend effect (slightly higher on weekends)
        weekend_effect = np.where(is_weekend, 1.1, 1.0)
        
        # Population density (moderate density optimal)
        pop_effect = np.exp(-((np.log(population_density + 1) - 3) ** 2) / 8)
        
        # Military proximity (higher near military bases)
        military_effect = 1 + 0.3 * np.exp(-military_proximity / 30)
        
        # Weather effect (higher probability in clear weather)
        weather_effect = np.where(weather_clear, 1.2, 0.8)
        
        # Combine all effects
        probability = (base_prob + hour_effect + lunar_effect) * \
                     seasonal_multiplier * weekend_effect * pop_effect * \
                     military_effect * weather_effect
        
        return np.clip(probability, 0, 1)