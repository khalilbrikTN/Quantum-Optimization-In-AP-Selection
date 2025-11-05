# OpenJij Benchmark Notebook

## Overview

The `OpenJij_Benchmark.ipynb` notebook provides a comprehensive benchmarking framework for the OpenJij Simulated Quantum Annealing (SQA) solver applied to the Access Point (AP) selection problem.

## Features

### 1. Configuration-Driven Experiments
- Single cell to configure all parameters
- Easy to modify and re-run with different settings
- Supports all three buildings (0, 1, 2)

### 2. Comprehensive Benchmarking
The notebook performs four types of benchmarks:

1. **Varying num_reads** (fixed num_sweeps)
   - Tests different sampling counts
   - Measures impact on solution quality and time

2. **Varying num_sweeps** (fixed num_reads)
   - Tests different annealing durations
   - Evaluates convergence behavior

3. **Parameter Grid Search** (num_reads × num_sweeps)
   - Explores parameter space systematically
   - Identifies optimal configurations

4. **Solution Quality Evaluation**
   - Trains ML models with selected APs
   - Measures localization accuracy (3D error, floor accuracy)

### 3. Metrics Tracked

#### Solver Performance
- **Time to Solution (TTS)**: Computation time for each configuration
- **Energy**: QUBO objective function value
- **Constraint Satisfaction**: Whether exactly k APs were selected

#### Solution Quality
- **Mean 3D Error**: Average localization error in meters
- **Median 3D Error**: Robust error metric
- **Floor Accuracy**: Percentage of correct floor predictions

### 4. Visualizations

The notebook generates comprehensive visualizations:
- Time vs. num_reads and num_sweeps
- Energy convergence plots
- Parameter grid heatmaps
- Time-energy trade-off scatter plots
- Solution quality vs. computation time

## Configuration Parameters

### QUBO Parameters
- `BUILDING_ID`: Which building to test (0, 1, or 2)
- `K`: Number of APs to select
- `ALPHA`: Importance vs. redundancy weight (0.0 to 1.0)
- `PENALTY`: Constraint penalty multiplier
- `IMPORTANCE_METHOD`: Which importance metric to use

### Benchmark Ranges
- `NUM_READS_RANGE`: List of num_reads values to test
- `NUM_SWEEPS_RANGE`: List of num_sweeps values to test
- `NUM_REPETITIONS`: How many times to repeat each experiment

## Usage

1. **Basic Usage**:
   ```python
   # Set parameters in configuration cell
   BUILDING_ID = 1
   K = 20
   ALPHA = 0.9

   # Run all cells
   ```

2. **Quick Test** (fewer experiments):
   ```python
   NUM_READS_RANGE = [50, 100, 200]
   NUM_SWEEPS_RANGE = [500, 1000]
   NUM_REPETITIONS = 3
   ```

3. **Comprehensive Benchmark** (more data points):
   ```python
   NUM_READS_RANGE = [10, 50, 100, 200, 500, 1000, 2000]
   NUM_SWEEPS_RANGE = [100, 500, 1000, 2000, 5000, 10000]
   NUM_REPETITIONS = 20
   ```

## Output Files

All results are saved to `data/results/benchmarks/`:

### CSV Files
- `benchmark_reads_<timestamp>.csv`: Results for varying num_reads
- `benchmark_sweeps_<timestamp>.csv`: Results for varying num_sweeps
- `benchmark_grid_<timestamp>.csv`: Results for parameter grid
- `benchmark_quality_<timestamp>.csv`: Solution quality evaluation
- `benchmark_summary_<timestamp>.csv`: Summary statistics

### Visualizations
- `time_to_solution.png`: Time analysis
- `energy_analysis.png`: Energy convergence
- `parameter_grid_heatmaps.png`: Heatmaps for all metrics
- `time_energy_tradeoff.png`: Pareto frontier
- `quality_vs_time.png`: Quality metrics vs. computation time

## Recommendations

Based on typical benchmark results:

### For Research/Offline Processing
- **num_reads**: 200-500
- **num_sweeps**: 2000-5000
- **Goal**: Best possible solution quality
- **Time**: ~30-60 seconds per run

### For Real-time/Online Applications
- **num_reads**: 50-100
- **num_sweeps**: 500-1000
- **Goal**: Fast results with acceptable quality
- **Time**: ~5-10 seconds per run

### Balanced Configuration
- **num_reads**: 100
- **num_sweeps**: 1000
- **Goal**: Good quality with reasonable time
- **Time**: ~10-15 seconds per run

## Tips

1. **Start Small**: Begin with small parameter ranges to estimate total runtime
2. **Statistical Significance**: Use NUM_REPETITIONS ≥ 10 for reliable statistics
3. **Memory**: Large grid searches can use significant memory; monitor usage
4. **Reproducibility**: Set RANDOM_SEED for reproducible results
5. **Comparison**: Run benchmarks on different buildings to compare difficulty

## Dependencies

Required packages:
- openjij
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- scikit-learn

All custom modules from the `scripts/` directory.

## References

- OpenJij Documentation: https://openjij.github.io/OpenJij/
- OpenJij Tutorial on Evaluation: https://tutorial.openjij.org/en/tutorial/005-Evaluation.html
