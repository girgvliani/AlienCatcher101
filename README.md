# Fully Connected Feedforward Neural Network (FC-FFNN) Project

## Overview
This project implements and investigates Fully Connected Feedforward Neural Networks for function approximation and real-world applications, with a focus on understanding the Universal Approximation Theorem and its practical implications.

## Project Structure
```
FC-FFNN-Project/
├── src/
│   ├── ffnn.py              # Main FC-FFNN implementation
│   ├── training.py          # Training algorithms and utilities
│   ├── evaluation.py        # Model evaluation and metrics
│   └── visualization.py     # Plotting and visualization tools
├── data/
│   ├── synthetic/           # Generated synthetic datasets
│   ├── real_world/          # Real-world application datasets
│   └── alien_sightings/     # UFO/alien sighting data
├── experiments/
│   ├── architecture_study/  # Network architecture experiments
│   ├── approximation_quality/ # Function approximation analysis
│   └── applications/        # Real-world application results
├── docs/
│   ├── theory.md           # Universal Approximation Theorem explanation
│   ├── architecture.md     # Network design principles
│   └── results.md          # Experimental results and analysis
└── README.md

```

## Tasks Completed

### 1. Universal Approximation Theorem Analysis
- **Theoretical Foundation**: Detailed explanation of the theorem and its implications
- **Practical Limitations**: Analysis of what the theorem doesn't address
- **Implementation**: Empirical validation through various function approximation tasks

### 2. FC-FFNN Implementation
- Pure Python implementation with NumPy
- Modular design for easy experimentation
- Support for different activation functions (ReLU, Sigmoid, Tanh)
- Configurable network architectures

### 3. Architecture Impact Investigation
- Systematic study of network depth vs. width
- Analysis of approximation quality vs. network complexity
- Performance comparison across different architectures

### 4. Approximation Quality Factors
- **Data Quality**: Impact of noise, outliers, and sampling density
- **Network Architecture**: Hidden layers, neurons per layer, activation functions
- **Training Methodology**: Optimization algorithms, learning rates, regularization
- **Hyperparameter Tuning**: Grid search and random search strategies

### 5. Real-World Applications

#### Alien Sightings Pattern Analysis
- **Data Source**: National UFO Reporting Center (NUFORC) database
- **Objective**: Predict sighting likelihood based on temporal and geographical features
- **Features**: Date, time, location, weather conditions, population density
- **Challenges**: Sparse data, reporting bias, classification ambiguity

#### Other Applications
- **Financial Prediction**: Stock price movement forecasting
- **Medical Diagnosis**: Disease classification from symptoms
- **Environmental Monitoring**: Air quality prediction

## Key Findings

### Universal Approximation Theorem Insights
- The theorem guarantees existence but not constructibility
- Practical approximation requires careful architecture design
- Training dynamics significantly impact achievable approximation quality

### Architecture Impact
- Deeper networks often achieve better approximation with fewer total parameters
- Width vs. depth trade-offs depend on function complexity
- Skip connections can improve approximation of complex functions

### Quality Factors
1. **Data preprocessing** has the largest impact on final performance
2. **Regularization techniques** are crucial for generalization
3. **Optimization algorithm choice** affects convergence speed and final accuracy
4. **Learning rate scheduling** improves approximation stability

## Installation and Usage

### Requirements
```bash
pip install numpy matplotlib scikit-learn pandas seaborn
```

### Basic Usage
```python
from src.ffnn import FCFFNN
from src.training import train_network

# Create and train network
model = FCFFNN(input_size=10, hidden_sizes=[64, 32], output_size=1)
trained_model = train_network(model, X_train, y_train)

# Evaluate performance
from src.evaluation import evaluate_model
results = evaluate_model(trained_model, X_test, y_test)
```

### Running Experiments
```bash
# Architecture comparison
python experiments/architecture_study/compare_architectures.py

# Function approximation analysis
python experiments/approximation_quality/analyze_approximation.py

# Alien sightings application
python experiments/applications/alien_sightings_predictor.py
```

## Results and Visualizations

### Function Approximation Results
- Successfully approximated various mathematical functions
- Demonstrated impact of network architecture on approximation quality
- Validated theoretical predictions with empirical evidence

### Alien Sightings Application Results
- Achieved 73% accuracy in predicting high-activity sighting periods
- Identified temporal patterns in UFO reporting
- Discovered correlation between weather conditions and sighting reports

## Future Work
- Implement advanced architectures (ResNet, DenseNet)
- Explore automated architecture search
- Apply to additional real-world domains
- Investigate interpretability techniques

## References
1. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks
2. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function
3. Universal approximation bounds for superpositions of a sigmoidal function

## License
MIT License - see LICENSE file for details

## Contributors
- [Nikoloz Girgvliani] - Project Creator