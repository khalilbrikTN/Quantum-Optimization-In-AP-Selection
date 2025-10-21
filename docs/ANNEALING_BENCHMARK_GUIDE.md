# Annealing Benchmarking Guide

## Overview

This guide explains how to use the new annealing benchmarking tools created for evaluating OpenJij and D-Wave Simulated Annealing performance.

**New Files Created:**
- `annealing_benchmark.py`: Core benchmarking module with analysis functions
- `Visualizations_Functions.py`: Enhanced with 9 new annealing-specific plotting functions

---

## Table of Contents

1. [Core Benchmarking Functions](#core-benchmarking-functions)
2. [Visualization Functions](#visualization-functions)
3. [Quick Start Examples](#quick-start-examples)
4. [Advanced Usage](#advanced-usage)
5. [Interpreting Results](#interpreting-results)

---

## Core Benchmarking Functions

### 1. `compare_solvers()` - **Main Entry Point**

Runs comprehensive comparison between OpenJij and D-Wave SA.

```python
from code_resources.annealing_benchmark import compare_solvers
from code_resources.QUBO import formulate_qubo

# Formulate your QUBO
Q, relevant_aps, offset = formulate_qubo(
    importance_scores,
    redundancy_matrix,
    k=20,
    alpha=0.7,
    penalty=100
)

# Benchmark both solvers
results = compare_solvers(
    qubo_model=Q,
    num_sweeps_range=[100, 500, 1000, 2000, 5000],
    num_reads=1000,
    num_trials=10,
    target_probability=0.99,
    seed=42
)

# Results structure:
# {
#   'openjij': DataFrame with benchmark metrics,
#   'dwave_sa': DataFrame with benchmark metrics
# }
```

**Output DataFrame Columns:**
- `num_sweeps`: Number of annealing sweeps tested
- `avg_time`: Average execution time (seconds)
- `std_time`: Standard deviation of time
- `best_energy`: Best (minimum) energy found
- `success_prob`: Probability of finding best energy
- `tts`: Time-to-Solution (99% success probability)
- `mean_residual`: Mean residual energy
- `std_residual`: Std deviation of residual energy
- `energy_mean`, `energy_std`, `energy_min`, `energy_max`: Energy statistics
- `energy_q90`: 90th percentile energy
- `energy_cv`: Coefficient of variation

---

### 2. `benchmark_openjij_sweep()` - OpenJij Only

Benchmark only OpenJij across different num_sweeps values.

```python
from code_resources.annealing_benchmark import benchmark_openjij_sweep

results_df = benchmark_openjij_sweep(
    qubo_model=Q,
    num_sweeps_range=[100, 500, 1000, 2000, 5000],
    num_reads=1000,
    num_trials=10,
    target_probability=0.99
)

print(results_df)
```

---

### 3. `benchmark_dwave_sa_sweep()` - D-Wave SA Only

Benchmark only D-Wave SA across different num_sweeps values.

```python
from code_resources.annealing_benchmark import benchmark_dwave_sa_sweep

results_df = benchmark_dwave_sa_sweep(
    qubo_model=Q,
    num_sweeps_range=[100, 500, 1000, 2000, 5000],
    num_reads=1000,
    num_trials=10,
    target_probability=0.99,
    seed=42
)

print(results_df)
```

---

### 4. `convergence_analysis()` - Energy Convergence Study

Analyze how solution quality improves with longer annealing times.

```python
from code_resources.annealing_benchmark import convergence_analysis

# For OpenJij
convergence_df = convergence_analysis(
    qubo_model=Q,
    solver_type='openjij',
    num_sweeps_progression=[10, 50, 100, 200, 500, 1000, 2000, 5000],
    num_reads=100
)

# For D-Wave SA
convergence_df = convergence_analysis(
    qubo_model=Q,
    solver_type='dwave_sa',
    num_sweeps_progression=[10, 50, 100, 200, 500, 1000, 2000, 5000],
    num_reads=100,
    seed=42
)
```

**Output Columns:**
- `num_sweeps`: Annealing duration
- `best_energy`: Best energy found
- `mean_energy`: Mean energy across runs
- `median_energy`: Median energy
- `std_energy`: Standard deviation
- `q90_energy`: 90th percentile
- `time`: Execution time

---

### 5. `analyze_solution_consistency()` - Solution Diversity

Understand how diverse the solutions are across multiple runs.

```python
from code_resources.annealing_benchmark import analyze_solution_consistency

consistency = analyze_solution_consistency(
    qubo_model=Q,
    solver_type='openjij',  # or 'dwave_sa'
    num_sweeps=1000,
    num_reads=100,
    num_trials=20,
    seed=42
)

print(f"Unique solutions: {consistency['unique_solutions']}")
print(f"Diversity score: {consistency['diversity_score']:.4f}")
print(f"Most frequent solution: {consistency['most_frequent_solution']}")
```

**Output Dictionary:**
- `unique_solutions`: Number of unique solutions found
- `total_samples`: Total number of samples
- `solution_frequencies`: Counter object with solution counts
- `energy_landscape`: Energy for each unique solution
- `most_frequent_solution`: Most common solution
- `most_frequent_count`: How often it appeared
- `diversity_score`: Ratio of unique solutions to total samples
- `all_energies`: All energy values collected

---

## Visualization Functions

### 1. `plot_tts_comparison()` - Time-to-Solution Plot

```python
from code_resources.Visualizations_Functions import plot_tts_comparison

plot_tts_comparison(
    results_dict=results,  # From compare_solvers()
    save_path='figures/tts_comparison.png'
)
```

**Shows:** Log-log plot of TTS vs. num_sweeps for both solvers.

---

### 2. `plot_success_probability()` - Success Rate Plot

```python
from code_resources.Visualizations_Functions import plot_success_probability

plot_success_probability(
    results_dict=results,
    save_path='figures/success_prob.png'
)
```

**Shows:** How success probability improves with more sweeps.

---

### 3. `plot_energy_convergence()` - Convergence Plot

```python
from code_resources.Visualizations_Functions import plot_energy_convergence

plot_energy_convergence(
    convergence_df=convergence_df,  # From convergence_analysis()
    solver_name='OpenJij SQA',
    save_path='figures/convergence_openjij.png'
)
```

**Shows:** Best, mean, and 90th percentile energy vs. num_sweeps.

---

### 4. `plot_residual_energy()` - Residual Energy Plot

```python
from code_resources.Visualizations_Functions import plot_residual_energy

plot_residual_energy(
    results_dict=results,
    save_path='figures/residual_energy.png'
)
```

**Shows:** Mean residual energy with error bars for both solvers.

---

### 5. `plot_energy_distribution()` - Histogram

```python
from code_resources.Visualizations_Functions import plot_energy_distribution

# Get energies from consistency analysis
plot_energy_distribution(
    energies=consistency['all_energies'],
    solver_name='OpenJij',
    bins=50,
    save_path='figures/energy_dist.png'
)
```

**Shows:** Distribution of energies with best, mean, and median markers.

---

### 6. `plot_solver_comparison_metrics()` - 2x2 Comparison

```python
from code_resources.Visualizations_Functions import plot_solver_comparison_metrics

plot_solver_comparison_metrics(
    results_dict=results,
    save_path='figures/comparison_metrics.png'
)
```

**Shows:** 4 subplots comparing success probability, time, energy, and residual.

---

### 7. `plot_solution_diversity()` - Diversity Analysis

```python
from code_resources.Visualizations_Functions import plot_solution_diversity

plot_solution_diversity(
    consistency_results=consistency,  # From analyze_solution_consistency()
    save_path='figures/solution_diversity.png'
)
```

**Shows:** Frequency and energy of top 20 most common solutions.

---

### 8. `plot_time_vs_quality_tradeoff()` - Trade-off Analysis

```python
from code_resources.Visualizations_Functions import plot_time_vs_quality_tradeoff

plot_time_vs_quality_tradeoff(
    results_dict=results,
    save_path='figures/time_quality_tradeoff.png'
)
```

**Shows:** Scatter plots showing computation time vs. solution quality.

---

## Quick Start Examples

### Example 1: Complete Benchmark and Visualization

```python
import sys
sys.path.append('code_resources')

from annealing_benchmark import compare_solvers
from Visualizations_Functions import (
    plot_tts_comparison,
    plot_success_probability,
    plot_solver_comparison_metrics
)
from QUBO import formulate_qubo

# 1. Load your data
# (assume importance_scores and redundancy_matrix are loaded)

# 2. Formulate QUBO
Q, relevant_aps, offset = formulate_qubo(
    importance_scores=importance_scores,
    redundancy_matrix=redundancy_matrix,
    k=20,
    alpha=0.7,
    penalty=100
)

# 3. Run benchmark
results = compare_solvers(
    qubo_model=Q,
    num_sweeps_range=[100, 500, 1000, 2000],
    num_reads=500,
    num_trials=5
)

# 4. Visualize results
plot_tts_comparison(results)
plot_success_probability(results)
plot_solver_comparison_metrics(results)

# 5. Export results to Excel
results['openjij'].to_excel('openjij_benchmark.xlsx', index=False)
results['dwave_sa'].to_excel('dwave_sa_benchmark.xlsx', index=False)
```

---

### Example 2: Find Optimal num_sweeps

```python
from annealing_benchmark import benchmark_openjij_sweep
import matplotlib.pyplot as plt

# Test different num_sweeps values
results_df = benchmark_openjij_sweep(
    qubo_model=Q,
    num_sweeps_range=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
    num_reads=1000,
    num_trials=10
)

# Find optimal num_sweeps (minimum TTS)
optimal_idx = results_df['tts'].idxmin()
optimal_sweeps = results_df.loc[optimal_idx, 'num_sweeps']
optimal_tts = results_df.loc[optimal_idx, 'tts']

print(f"Optimal num_sweeps: {optimal_sweeps}")
print(f"TTS: {optimal_tts:.2f} seconds")

# Plot TTS vs num_sweeps
plt.figure(figsize=(10, 6))
plt.plot(results_df['num_sweeps'], results_df['tts'], 'o-', linewidth=2)
plt.axvline(optimal_sweeps, color='red', linestyle='--',
            label=f'Optimal: {optimal_sweeps}')
plt.xlabel('Number of Sweeps', fontsize=14)
plt.ylabel('Time-to-Solution (s)', fontsize=14)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Example 3: Solution Consistency Analysis

```python
from annealing_benchmark import analyze_solution_consistency
from Visualizations_Functions import plot_solution_diversity

# Analyze OpenJij
consistency_openjij = analyze_solution_consistency(
    qubo_model=Q,
    solver_type='openjij',
    num_sweeps=1000,
    num_reads=100,
    num_trials=20
)

# Analyze D-Wave SA
consistency_dwave = analyze_solution_consistency(
    qubo_model=Q,
    solver_type='dwave_sa',
    num_sweeps=1000,
    num_reads=100,
    num_trials=20,
    seed=42
)

# Compare diversity
print(f"OpenJij diversity: {consistency_openjij['diversity_score']:.4f}")
print(f"D-Wave SA diversity: {consistency_dwave['diversity_score']:.4f}")

# Visualize
plot_solution_diversity(consistency_openjij)
plot_solution_diversity(consistency_dwave)
```

---

## Advanced Usage

### Custom Benchmark with Multiple k Values

```python
from annealing_benchmark import compare_solvers
import pandas as pd

k_values = [10, 15, 20, 25, 30]
all_results = []

for k in k_values:
    print(f"\n{'='*60}")
    print(f"Benchmarking k={k}")
    print(f"{'='*60}")

    # Formulate QUBO for this k
    Q, relevant_aps, offset = formulate_qubo(
        importance_scores, redundancy_matrix, k, alpha=0.7, penalty=100
    )

    # Benchmark
    results = compare_solvers(
        qubo_model=Q,
        num_sweeps_range=[100, 500, 1000, 2000],
        num_reads=500,
        num_trials=5
    )

    # Add k to results
    results['openjij']['k'] = k
    results['dwave_sa']['k'] = k

    all_results.append({
        'k': k,
        'openjij': results['openjij'],
        'dwave_sa': results['dwave_sa']
    })

# Combine all results
all_openjij = pd.concat([r['openjij'] for r in all_results], ignore_index=True)
all_dwave = pd.concat([r['dwave_sa'] for r in all_results], ignore_index=True)

# Save comprehensive results
all_openjij.to_excel('benchmark_openjij_all_k.xlsx', index=False)
all_dwave.to_excel('benchmark_dwave_all_k.xlsx', index=False)
```

---

## Interpreting Results

### Time-to-Solution (TTS)

- **Lower is better**: Indicates faster time to find optimal solution with 99% probability
- **Infinite TTS**: Means success probability was 0% at that num_sweeps
- **Interpretation**: If TTS decreases with more sweeps initially but then plateaus or increases, you've found the optimal annealing duration

### Success Probability

- **Range**: 0% to 100%
- **Target**: Aim for at least 80-90% for reliable results
- **Interpretation**: Shows how often the solver finds the best energy. Low success probability means the annealing is trapped in local minima.

### Residual Energy

- **Definition**: Difference between average energy obtained and best energy
- **Range**: 0 (perfect) to positive values (worse)
- **Interpretation**: Measures how far typical solutions are from the best solution. Lower residual = more consistent solver.

### Energy Coefficient of Variation (CV)

- **Definition**: Std deviation / Mean energy
- **Interpretation**: Measures relative variability. Lower CV = more consistent results across runs.

### Diversity Score

- **Range**: 0 (all identical) to 1.0 (all unique)
- **High diversity (>0.5)**: Solver explores many different solutions (may indicate weak energy landscape)
- **Low diversity (<0.1)**: Solver consistently finds same solutions (good if they're optimal, bad if suboptimal)

---

## Best Practices

1. **Start with wide num_sweeps range**: `[100, 500, 1000, 2000, 5000, 10000]`
2. **Use enough trials**: At least 5-10 trials for statistical reliability
3. **Balance num_reads and num_trials**: More reads = better statistics per trial, more trials = better overall confidence
4. **Save results**: Always export to Excel/CSV for later analysis
5. **Visualize everything**: Use all visualization functions to get complete picture
6. **Compare relative performance**: Focus on which solver is *better* rather than absolute values
7. **Consider problem size**: Larger QUBO problems (higher k) may need more sweeps

---

## Integration with Existing Pipeline

To integrate benchmarking into your existing workflow:

```python
# In your RUNNER.ipynb or analysis script

# After formulating QUBO
Q, relevant_aps, offset = formulate_qubo(...)

# BEFORE using solve_qubo_with_openjij() or solve_qubo_with_SA()
# Run benchmark to find optimal parameters
from code_resources.annealing_benchmark import compare_solvers

results = compare_solvers(
    qubo_model=Q,
    num_sweeps_range=[100, 500, 1000, 2000, 5000],
    num_reads=1000,
    num_trials=5
)

# Find optimal num_sweeps for each solver
optimal_sweeps_openjij = results['openjij'].loc[
    results['openjij']['tts'].idxmin(), 'num_sweeps'
]
optimal_sweeps_dwave = results['dwave_sa'].loc[
    results['dwave_sa']['tts'].idxmin(), 'num_sweeps'
]

# NOW solve with optimal parameters
selected_indices_openjij, duration_openjij = solve_qubo_with_openjij(
    Q, num_reads=1000, num_sweeps=int(optimal_sweeps_openjij)
)

selected_indices_dwave, duration_dwave = solve_qubo_with_SA(
    Q, num_reads=1000, num_sweeps=int(optimal_sweeps_dwave)
)

# Continue with ML training, evaluation, etc.
```

---

## Questions?

If you need help or encounter issues:

1. Check function docstrings: `help(compare_solvers)`
2. Review the OpenJij tutorial: https://tutorial.openjij.org/en/tutorial/005-Evaluation.html
3. Examine example outputs in this guide

---

## Summary of New Capabilities

✅ **Time-to-Solution (TTS)** calculation for both solvers
✅ **Success probability** analysis across different annealing durations
✅ **Residual energy** tracking and comparison
✅ **Energy distribution** analysis and visualization
✅ **Solution consistency** and diversity metrics
✅ **Convergence analysis** showing quality improvement over time
✅ **Comprehensive visualizations** (9 new plotting functions)
✅ **Side-by-side solver comparison** with statistical confidence
✅ **Parameter optimization** to find best num_sweeps

**Result:** You can now scientifically justify your choice of solver and annealing parameters in your thesis!
