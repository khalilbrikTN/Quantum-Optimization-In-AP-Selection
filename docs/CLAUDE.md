# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for WiFi-based indoor localization using quantum-inspired optimization (QUBO formulation) for Access Point (AP) selection. The system uses the UJIIndoorLoc dataset to predict 3D coordinates (longitude, latitude, floor) from WiFi RSSI (Received Signal Strength Indicator) measurements.

The core research question: Which subset of WiFi Access Points should be selected to maximize localization accuracy while minimizing redundancy?

## High-Level Architecture

The codebase follows a **pipeline architecture** with these key stages:

1. **Data Preprocessing** → 2. **Feature Importance Calculation** → 3. **Redundancy Analysis** → 4. **QUBO Formulation** → 5. **Quantum-Inspired Solving** → 6. **ML Training** → 7. **Performance Evaluation** → 8. **Visualization**

### Core Pipeline Flow

```
Training/Validation CSV → pre_processing.py → Normalized RSSI + 3D Coords
                                ↓
                        Importance.py (5 methods) → Importance Scores
                                ↓
                        Redundancy.py → Correlation Matrix
                                ↓
                        QUBO.py → QUBO Matrix (Q)
                                ↓
                Two Solvers: OpenJij (SQA) / D-Wave (SA) → Selected APs
                                ↓
                        ML_post_processing.py → Random Forest Regressor
                                ↓
                        Analysis.py → 3D Error Metrics + Floor Accuracy
                                ↓
                        Visualizations_Functions.py → Plots
```

### Key Module Responsibilities

- **`pre_processing.py`**: Loads UJIIndoorLoc data, filters by building, normalizes RSSI to [0,1], normalizes coordinates, stores global normalization constants (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, FLOOR_HEIGHT)

- **`Importance.py`**: Five importance metrics (entropy, average, median, max, variance) to score each AP's localization value

- **`Redundancy.py`**: Calculates Pearson correlation matrix between all APs to identify redundant signals

- **`QUBO.py`**:
  - `formulate_qubo()`: Creates QUBO matrix balancing importance maximization vs redundancy minimization with constraint penalty for selecting exactly k APs
  - `solve_qubo_with_openjij()`: OpenJij Simulated Quantum Annealing solver
  - `solve_qubo_with_SA()`: D-Wave Simulated Annealing solver

- **`ML_post_processing.py`**: Trains Random Forest multi-output regressor on selected APs to predict (LON_NORM, LAT_NORM, FLOOR)

- **`Analysis.py`**:
  - Denormalizes predictions back to real-world coordinates
  - Calculates 3D Euclidean distance errors in meters
  - Computes floor accuracy (percentage of correctly predicted floors)
  - Handles UTM coordinate system properly (latitude/longitude in meters)

- **`run_exps.py`**: Orchestrates multi-budget experiments, running the full pipeline for different k values (number of APs), with Excel export functionality

- **`Visualizations_Functions.py`**: Plotting functions for error comparisons, floor accuracy, CDF plots, radar charts

- **`helpers.py`**: Utility for saving DataFrames to Excel with sheet management

- **`RUNNER.ipynb`**: Interactive notebook demonstrating the workflow

## Data Structure

### Input Data
- **Location**: `data/input_data/`
- **Files**: `TrainingData.csv`, `ValidationData.csv`
- **Format**: UJIIndoorLoc dataset with 520 WAP (WiFi Access Point) columns + coordinate columns (LATITUDE, LONGITUDE, FLOOR, BUILDINGID)
- **RSSI Encoding**: Value 100 = not detected (converted to NaN during preprocessing)

### Output Data
- **Importance Scores**: `data/output_data/importance_scores/` (Excel files per method)
- **Redundancy Matrix**: `data/output_data/redundancy_scores/redundancy_matrix.xlsx`

## Key Concepts

### QUBO Formulation
The QUBO (Quadratic Unconstrained Binary Optimization) objective balances three terms:

```
Minimize: -α * Σ(importance_i * x_i) + (1-α) * ΣΣ(redundancy_ij * x_i * x_j) + penalty * (Σx_i - k)²
```

- **α** (alpha): Weight parameter [0,1] balancing importance vs redundancy
- **penalty**: Constraint violation penalty for selecting exactly k APs
- **x_i**: Binary variable (1 if AP i is selected, 0 otherwise)
- **Adaptive penalty**: Scales with problem size (n_aps / 100.0)
- **Redundancy threshold**: 0.3 (only penalizes highly correlated APs)

### Coordinate System
- **Latitude/Longitude**: UTM coordinates in meters (not degrees)
- **Floor**: Integer values (0, 1, 2, 3, ...)
- **Floor Height**: Typically 3.0 meters per floor
- **3D Distance Calculation**: √(Δlat² + Δlon² + (Δfloor * floor_height)²)

### Performance Metrics
- **Mean/Median 3D Error**: Average Euclidean distance error in meters
- **Floor Accuracy**: Percentage of correctly predicted floors
- **90th/95th Percentile Error**: Error thresholds covering 90%/95% of predictions
- **Overfitting Ratio**: Validation error / Training error

## Common Development Commands

### Running Experiments

**Single budget experiment** (interactive):
```python
# In RUNNER.ipynb or Python script
from run_exps import run_multi_budget_experiment

results = run_multi_budget_experiment(
    df_train_path='data/input_data/TrainingData.csv',
    df_validation_path='data/input_data/ValidationData.csv',
    building_id=1,
    floor_height=3.0,
    budget_list=[10, 20, 30],  # k values to test
    alpha=0.9,
    penalty=2.0,
    num_reads=1000,
    num_sweeps=1000,
    output_folder='data/output_data/experiment_results'
)
```

### Data Preprocessing

```python
from pre_processing import load_and_preprocess_data

rssi_train, coords_train, rssi_val, coords_val, ap_columns = load_and_preprocess_data(
    'data/input_data/TrainingData.csv',
    'data/input_data/ValidationData.csv',
    building_id=1,
    floor_height=3.0
)
```

### Calculating Importance

```python
from Importance import entropy_importance, average_importance

# Returns (array, dict)
importance_array, importance_dict = entropy_importance(rssi_train, ap_columns)
```

### QUBO Solving

```python
from QUBO import formulate_qubo, solve_qubo_with_openjij

Q, relevant_aps, offset = formulate_qubo(
    importance_dict, redundancy_matrix, k=20, alpha=0.9, penalty=2.0
)

selected_indices, duration = solve_qubo_with_openjij(Q, num_reads=1000, num_sweeps=1000)
selected_aps = [relevant_aps[i] for i in selected_indices]
```

### Training Models

```python
from ML_post_processing import train_regressor

models, predictions = train_regressor(
    rssi_train, coords_train, rssi_val, coords_val,
    selected_aps, model_type='random_forest'
)

# predictions contains: {'rf_train': ..., 'rf_val': ...}
```

### Evaluation

```python
from Analysis import calculate_comprehensive_metrics
from pre_processing import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, FLOOR_HEIGHT

norm_errors, real_errors, metrics = calculate_comprehensive_metrics(
    coords_val, predictions['rf_val'],
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, FLOOR_HEIGHT
)

print(f"Mean 3D Error: {metrics['real_mean_m']:.2f} meters")
print(f"Floor Accuracy: {metrics['floor_accuracy']:.1%}")
```

## Important Design Patterns

### Global State in pre_processing.py
After running `load_and_preprocess_data()`, these globals are set and used by Analysis.py:
- `LAT_MIN`, `LAT_MAX`
- `LON_MIN`, `LON_MAX`
- `FLOOR_HEIGHT`

**Important**: Always call `load_and_preprocess_data()` before evaluation functions that denormalize coordinates.

### Multi-Output Regression
The system predicts 3D coordinates simultaneously:
```python
coords_train.shape  # (n_samples, 3) → [LON_NORM, LAT_NORM, FLOOR]
```

Random Forest uses `MultiOutputRegressor` wrapper for this.

### Importance Dictionary Structure
All importance methods return consistent format:
```python
{'WAP001': 0.523, 'WAP002': 0.0, 'WAP003': 1.245, ...}
```

Only APs with importance > 0 are used in QUBO formulation.

## Dependencies

Key libraries:
- `openjij`: OpenJij Simulated Quantum Annealing
- `dwave-ocean-sdk`, `dimod`: D-Wave tools for QUBO/BQM
- `scikit-learn`: Random Forest, mutual information, metrics
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization

## Experimental Parameters

Typical parameter ranges based on the codebase:
- **k (budget)**: 10-50 APs (out of 520 total)
- **alpha**: 0.7-0.95 (higher = prioritize importance over redundancy reduction)
- **penalty**: 1.0-5.0 (constraint enforcement strength)
- **num_reads**: 1000 (annealing runs for solution quality)
- **num_sweeps**: 1000 (annealing duration per run)
- **floor_height**: 3.0 meters (building-specific)
- **building_id**: 0, 1, or 2 (UJIIndoorLoc has 3 buildings)

## File Naming Conventions

- **Python modules**: lowercase with underscores (`pre_processing.py`)
- **Classes/Important concepts**: CamelCase in documentation
- **AP columns**: Format `WAP###` (e.g., WAP001, WAP248)
- **Excel outputs**: Descriptive names with timestamps for experiments

## Jupyter Notebook Usage

The `RUNNER.ipynb` notebook demonstrates the complete workflow. When modifying:
- Cells 0-1: Install dependencies
- Cell 2: Import all modules
- Cell 4-6: Load and preprocess data
- Cells 8-10: Calculate importance scores
- Cells 13-14: Calculate redundancy
- Cells 17-20: Run QUBO experiments
- Cells 22-29: Visualize results

## Notes on Solver Performance

From the notebook outputs:
- **OpenJij**: 50-80 seconds for k=20, better for exploration
- **D-Wave SA**: 50-80 seconds for k=20, deterministic with seed
- Both produce comparable localization accuracy (15-20m mean error for k=20)
- Results depend heavily on importance metric choice (entropy, average, max, variance)