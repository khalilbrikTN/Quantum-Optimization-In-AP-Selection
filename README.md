# Quantum Optimization in Access Point Selection

A quantum-inspired optimization approach for WiFi Access Point (AP) selection in indoor positioning systems using QUBO (Quadratic Unconstrained Binary Optimization) formulation.

## Overview

This project tackles the challenge of selecting optimal WiFi Access Points for indoor localization using quantum annealing techniques. By combining importance metrics with redundancy analysis through QUBO formulation, we efficiently select a subset of APs that maximize positioning accuracy while minimizing computational overhead.

## Key Features

- **Multiple Importance Metrics**: Entropy, Average, Median, Max, Variance, and Mutual Information-based AP importance calculation
- **Redundancy Analysis**: Correlation-based redundancy matrix to avoid selecting similar APs
- **QUBO Formulation**: Combines importance scores and redundancy penalties into a quantum optimization problem
- **Quantum Solvers**:
  - OpenJij Simulated Quantum Annealing (SQA)
  - D-Wave Simulated Annealing
- **Machine Learning Integration**: Random Forest Regressor for position estimation
- **Comprehensive Evaluation**: 3D error metrics and floor accuracy analysis

## Project Structure

```
.
├── data/
│   ├── input_data/          # Training and validation datasets
│   ├── output_data/         # Computed importance scores, redundancy matrices
│   ├── system_input/        # System parameters (normalization values)
│   ├── results/             # Experiment results
│   └── EDA/                 # Exploratory data analysis outputs
│
├── scripts/
│   ├── data/                # Data preprocessing and loading utilities
│   ├── optimization/        # Importance, Redundancy, QUBO formulation
│   ├── ml/                  # Machine learning post-processing
│   ├── evaluation/          # Performance metrics calculation
│   ├── visualization/       # Plotting and visualization functions
│   ├── utils/               # Helper functions
│   └── experiments/         # Experiment runners and benchmarks
│
├── notebooks/
│   ├── 00_data_preprocessing/   # Data preprocessing notebooks
│   ├── 01_experiments/          # Main experiment runners
│   ├── 02_analysis/             # Analysis and EDA notebooks
│   └── 03_visualizations/       # Visualization notebooks
│
├── tests/                   # Unit tests and validation scripts
├── docs/                    # Documentation
│   ├── setup/               # Setup and installation guides
│   └── architecture/        # Architecture diagrams and presentations
│
├── requirements.txt         # Python dependencies
└── start_jupyter.bat        # Jupyter launcher script
```

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name quantum-venv --display-name "Python (Quantum VEnv)"
```

### 2. Run Experiments

**Option A: Using Jupyter Notebooks**

```bash
# Start Jupyter
jupyter notebook

# Or use the convenience script (Windows)
start_jupyter.bat
```

Open `notebooks/01_experiments/RUNNER.ipynb` and select the "Python (Quantum VEnv)" kernel.

**Option B: Using Python Scripts**

```bash
python scripts/experiments/run_exps.py
```

### 3. Verify Installation

```bash
python tests/test_full_setup.py
```

Expected output: `ALL TESTS PASSED!`

## Latest Results (k=20 APs from 520 total)

| Importance Method | Mean 3D Error | Median 3D Error | Floor Accuracy |
|-------------------|---------------|-----------------|----------------|
| Average           | 15.14m        | 11.38m          | 66.12%         |
| Entropy           | 15.99m        | 11.76m          | 58.96%         |
| Max               | 16.67m        | 12.68m          | 69.06%         |
| Variance          | 16.75m        | 12.13m          | 65.15%         |
| Mutual Info       | 19.65m        | 13.63m          | 62.87%         |

## Documentation

- **Quick Start**: `docs/setup/QUICK_START.md`
- **Virtual Environment Setup**: `docs/setup/VENV_SETUP_GUIDE.md`
- **Installation Troubleshooting**: `docs/setup/INSTALLATION_SUMMARY.md`
- **Jupyter Configuration**: `docs/setup/JUPYTER_FIX_GUIDE.md`
- **Path Fix Summary**: `docs/setup/PATH_FIX_SUMMARY.md`
- **Before/After Comparison**: `docs/setup/BEFORE_AND_AFTER.md`

## Dependencies

### Core Quantum Libraries
- `openjij` - Simulated Quantum Annealing
- `dwave-ocean-sdk` - D-Wave quantum computing tools
- `dimod` - Discrete optimization models

### Machine Learning & Data Science
- `scikit-learn` - Machine learning algorithms
- `numpy`, `pandas`, `scipy` - Data manipulation and analysis

### Visualization
- `matplotlib`, `seaborn` - Plotting and visualization

### Other
- `jupyter`, `notebook` - Interactive notebooks
- `openpyxl` - Excel file handling

## Dataset

The project uses the **UJIIndoorLoc** dataset:
- **Training**: 19,937 samples
- **Validation**: 1,111 samples
- **Buildings**: 3 buildings (experiments focus on Building 1)
- **APs**: 520 WiFi access points
- **Features**: RSSI values, coordinates (longitude, latitude, floor)

## Workflow

1. **Data Preprocessing**: Normalize RSSI values and coordinates
2. **Importance Calculation**: Compute AP importance using 6 different metrics
3. **Redundancy Analysis**: Calculate correlation-based redundancy matrix
4. **QUBO Formulation**: Combine importance and redundancy into optimization problem
5. **Quantum Solving**: Use quantum annealing to select optimal APs
6. **ML Training**: Train Random Forest on selected APs
7. **Evaluation**: Calculate positioning errors and floor accuracy

## Testing

Run all tests:

```bash
# Comprehensive setup test
python tests/test_full_setup.py

# OpenJij specific test
python tests/test_openjij.py

# Data loaders test
python tests/test_data_loaders.py

# Jupyter environment check
python tests/check_jupyter_env.py
```

## Contributing

This is a research project for quantum optimization in indoor positioning systems. For questions or collaboration, please open an issue.

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Acknowledgments

- UJIIndoorLoc dataset creators
- OpenJij development team
- D-Wave Systems
