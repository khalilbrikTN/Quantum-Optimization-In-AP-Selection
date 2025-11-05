"""
Annealing Benchmarking Module

This module provides comprehensive benchmarking and evaluation tools for
simulated annealing solvers (OpenJij and D-Wave SA) used in QUBO optimization.

Key Metrics Implemented:
1. Time-to-Solution (TTS): Computation time needed to find optimal solutions with target probability
2. Success Probability: Ratio of runs achieving the best known energy
3. Residual Energy: Difference between obtained energy and best known energy
4. Energy Distribution Analysis: Statistical analysis of energy landscapes across runs

Based on OpenJij tutorial: https://tutorial.openjij.org/en/tutorial/005-Evaluation.html
"""

import numpy as np
import pandas as pd
import time
import openjij as oj
from dwave.samplers import SimulatedAnnealingSampler
import dimod
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict


def calculate_time_to_solution(success_probability, avg_time, target_probability=0.99):
    """
    Calculate Time-to-Solution (TTS) metric.

    TTS represents the time needed to find an optimal solution with a specified
    target probability, accounting for multiple runs.

    Formula: TTS(τ, p_R) = τ × ln(1-p_R) / ln(1-p_s(τ))

    Parameters:
    -----------
    success_probability : float
        Probability of finding optimal solution in a single run (p_s)
    avg_time : float
        Average time per run (τ)
    target_probability : float, optional
        Desired probability of finding solution (p_R). Default: 0.99 (99%)

    Returns:
    --------
    float
        Time-to-Solution in same units as avg_time
        Returns np.inf if success_probability is 0 or 1
    """
    if success_probability == 0 or success_probability >= 1.0:
        return np.inf

    try:
        tts = avg_time * np.log(1 - target_probability) / np.log(1 - success_probability)
        return tts
    except (ValueError, ZeroDivisionError):
        return np.inf


def calculate_success_probability(energies, best_energy, tolerance=1e-6):
    """
    Calculate success probability: ratio of runs achieving best energy.

    Parameters:
    -----------
    energies : array-like
        Array of energy values from multiple runs
    best_energy : float
        Best (minimum) energy value to compare against
    tolerance : float, optional
        Tolerance for floating point comparison. Default: 1e-6

    Returns:
    --------
    float
        Success probability (0.0 to 1.0)
    """
    energies = np.array(energies)
    successful_runs = np.sum(np.abs(energies - best_energy) <= tolerance)
    return successful_runs / len(energies)


def calculate_residual_energy(energies, best_energy):
    """
    Calculate residual energy: difference between average energy and best energy.

    Parameters:
    -----------
    energies : array-like
        Array of energy values from multiple runs
    best_energy : float
        Best (minimum) energy value

    Returns:
    --------
    dict
        Dictionary containing:
        - 'mean_residual': Mean residual energy
        - 'std_residual': Standard deviation of residual
        - 'median_residual': Median residual energy
    """
    energies = np.array(energies)
    residuals = energies - best_energy

    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'median_residual': np.median(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals)
    }


def analyze_energy_distribution(energies):
    """
    Comprehensive statistical analysis of energy distribution.

    Parameters:
    -----------
    energies : array-like
        Array of energy values from multiple runs

    Returns:
    --------
    dict
        Dictionary containing statistical measures:
        - 'mean', 'std', 'median', 'min', 'max'
        - 'q25', 'q75': 25th and 75th percentiles
        - 'q90', 'q95', 'q99': 90th, 95th, 99th percentiles
        - 'iqr': Interquartile range
        - 'cv': Coefficient of variation (std/mean)
    """
    energies = np.array(energies)

    stats = {
        'mean': np.mean(energies),
        'std': np.std(energies),
        'median': np.median(energies),
        'min': np.min(energies),
        'max': np.max(energies),
        'q25': np.percentile(energies, 25),
        'q75': np.percentile(energies, 75),
        'q90': np.percentile(energies, 90),
        'q95': np.percentile(energies, 95),
        'q99': np.percentile(energies, 99),
    }

    stats['iqr'] = stats['q75'] - stats['q25']
    stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] != 0 else np.inf

    return stats


def benchmark_openjij_sweep(qubo_model, num_sweeps_range, num_reads=1000,
                            num_trials=10, target_probability=0.99):
    """
    Benchmark OpenJij solver across different num_sweeps values.

    Tests multiple annealing durations to find optimal parameter settings
    and calculates TTS, success probability, and residual energy.

    Parameters:
    -----------
    qubo_model : dict
        QUBO matrix as dictionary {(i,j): coefficient}
    num_sweeps_range : list or array
        List of num_sweeps values to test
    num_reads : int, optional
        Number of annealing runs per trial. Default: 1000
    num_trials : int, optional
        Number of independent trials for each num_sweeps. Default: 10
    target_probability : float, optional
        Target probability for TTS calculation. Default: 0.99

    Returns:
    --------
    pandas.DataFrame
        Results with columns: num_sweeps, avg_time, best_energy, success_prob,
        tts, mean_residual, std_residual, energy_stats
    """
    print("Benchmarking OpenJij across num_sweeps range...")
    results = []

    for num_sweeps in num_sweeps_range:
        print(f"  Testing num_sweeps={num_sweeps}...")

        trial_times = []
        trial_energies = []
        all_energies = []

        for trial in range(num_trials):
            sampler = oj.SQASampler()

            start_time = time.time()
            response = sampler.sample_qubo(
                qubo_model,
                num_reads=num_reads,
                num_sweeps=num_sweeps
            )
            end_time = time.time()

            trial_times.append(end_time - start_time)

            # Extract all energies from this trial
            energies = [sample.energy for sample in response.data(['sample', 'energy'])]
            all_energies.extend(energies)

            # Best energy from this trial
            trial_energies.append(response.first.energy)

        # Calculate metrics
        avg_time = np.mean(trial_times)
        best_energy = np.min(all_energies)

        success_prob = calculate_success_probability(all_energies, best_energy)
        tts = calculate_time_to_solution(success_prob, avg_time, target_probability)

        residual_stats = calculate_residual_energy(all_energies, best_energy)
        energy_stats = analyze_energy_distribution(all_energies)

        results.append({
            'num_sweeps': num_sweeps,
            'avg_time': avg_time,
            'std_time': np.std(trial_times),
            'best_energy': best_energy,
            'success_prob': success_prob,
            'tts': tts,
            'mean_residual': residual_stats['mean_residual'],
            'std_residual': residual_stats['std_residual'],
            'median_residual': residual_stats['median_residual'],
            'energy_mean': energy_stats['mean'],
            'energy_std': energy_stats['std'],
            'energy_min': energy_stats['min'],
            'energy_max': energy_stats['max'],
            'energy_q90': energy_stats['q90'],
            'energy_cv': energy_stats['cv']
        })

    df = pd.DataFrame(results)
    print("OpenJij benchmarking complete!")
    return df


def benchmark_dwave_sa_sweep(qubo_model, num_sweeps_range, num_reads=1000,
                              num_trials=10, target_probability=0.99, seed=42):
    """
    Benchmark D-Wave Simulated Annealing solver across different num_sweeps values.

    Tests multiple annealing durations to find optimal parameter settings
    and calculates TTS, success probability, and residual energy.

    Parameters:
    -----------
    qubo_model : dict
        QUBO matrix as dictionary {(i,j): coefficient}
    num_sweeps_range : list or array
        List of num_sweeps values to test
    num_reads : int, optional
        Number of annealing runs per trial. Default: 1000
    num_trials : int, optional
        Number of independent trials for each num_sweeps. Default: 10
    target_probability : float, optional
        Target probability for TTS calculation. Default: 0.99
    seed : int, optional
        Random seed for reproducibility. Default: 42

    Returns:
    --------
    pandas.DataFrame
        Results with columns: num_sweeps, avg_time, best_energy, success_prob,
        tts, mean_residual, std_residual, energy_stats
    """
    print("Benchmarking D-Wave SA across num_sweeps range...")
    results = []

    for num_sweeps in num_sweeps_range:
        print(f"  Testing num_sweeps={num_sweeps}...")

        trial_times = []
        trial_energies = []
        all_energies = []

        for trial in range(num_trials):
            bqm = dimod.BinaryQuadraticModel(qubo_model, 'BINARY')
            sampler = SimulatedAnnealingSampler()

            start_time = time.time()
            response = sampler.sample(
                bqm,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                beta_range=(0.1, 5.0),
                seed=seed + trial  # Different seed per trial
            )
            end_time = time.time()

            trial_times.append(end_time - start_time)

            # Extract all energies from this trial
            energies = [sample.energy for sample in response.data(['sample', 'energy'])]
            all_energies.extend(energies)

            # Best energy from this trial
            trial_energies.append(response.first.energy)

        # Calculate metrics
        avg_time = np.mean(trial_times)
        best_energy = np.min(all_energies)

        success_prob = calculate_success_probability(all_energies, best_energy)
        tts = calculate_time_to_solution(success_prob, avg_time, target_probability)

        residual_stats = calculate_residual_energy(all_energies, best_energy)
        energy_stats = analyze_energy_distribution(all_energies)

        results.append({
            'num_sweeps': num_sweeps,
            'avg_time': avg_time,
            'std_time': np.std(trial_times),
            'best_energy': best_energy,
            'success_prob': success_prob,
            'tts': tts,
            'mean_residual': residual_stats['mean_residual'],
            'std_residual': residual_stats['std_residual'],
            'median_residual': residual_stats['median_residual'],
            'energy_mean': energy_stats['mean'],
            'energy_std': energy_stats['std'],
            'energy_min': energy_stats['min'],
            'energy_max': energy_stats['max'],
            'energy_q90': energy_stats['q90'],
            'energy_cv': energy_stats['cv']
        })

    df = pd.DataFrame(results)
    print("D-Wave SA benchmarking complete!")
    return df


def compare_solvers(qubo_model, num_sweeps_range, num_reads=1000,
                   num_trials=10, target_probability=0.99, seed=42):
    """
    Compare OpenJij and D-Wave SA solvers side-by-side.

    Runs both benchmarking functions and returns combined results
    for easy comparison and visualization.

    Parameters:
    -----------
    qubo_model : dict
        QUBO matrix as dictionary {(i,j): coefficient}
    num_sweeps_range : list or array
        List of num_sweeps values to test
    num_reads : int, optional
        Number of annealing runs per trial. Default: 1000
    num_trials : int, optional
        Number of independent trials. Default: 10
    target_probability : float, optional
        Target probability for TTS calculation. Default: 0.99
    seed : int, optional
        Random seed for D-Wave SA. Default: 42

    Returns:
    --------
    dict
        Dictionary with keys 'openjij' and 'dwave_sa', each containing
        a DataFrame of benchmark results
    """
    print("="*60)
    print("COMPARATIVE SOLVER BENCHMARKING")
    print("="*60)

    # Benchmark OpenJij
    openjij_results = benchmark_openjij_sweep(
        qubo_model, num_sweeps_range, num_reads, num_trials, target_probability
    )
    openjij_results['solver'] = 'OpenJij'

    print()

    # Benchmark D-Wave SA
    dwave_results = benchmark_dwave_sa_sweep(
        qubo_model, num_sweeps_range, num_reads, num_trials, target_probability, seed
    )
    dwave_results['solver'] = 'D-Wave SA'

    print("="*60)
    print("BENCHMARKING COMPLETE")
    print("="*60)

    return {
        'openjij': openjij_results,
        'dwave_sa': dwave_results
    }


def analyze_solution_consistency(qubo_model, solver_type='openjij',
                                 num_sweeps=1000, num_reads=100,
                                 num_trials=20, seed=42):
    """
    Analyze consistency of solutions across multiple annealing runs.

    Helps understand solution diversity and repeatability.

    Parameters:
    -----------
    qubo_model : dict
        QUBO matrix as dictionary
    solver_type : str, optional
        Either 'openjij' or 'dwave_sa'. Default: 'openjij'
    num_sweeps : int, optional
        Number of annealing sweeps. Default: 1000
    num_reads : int, optional
        Number of reads per trial. Default: 100
    num_trials : int, optional
        Number of independent trials. Default: 20
    seed : int, optional
        Random seed (for D-Wave SA). Default: 42

    Returns:
    --------
    dict
        Dictionary containing:
        - 'unique_solutions': Number of unique solutions found
        - 'solution_frequencies': Counter of solution occurrences
        - 'energy_landscape': Energy values for each unique solution
        - 'most_frequent_solution': Most commonly found solution
        - 'diversity_score': Ratio of unique solutions to total samples
    """
    print(f"Analyzing solution consistency for {solver_type}...")

    all_solutions = []
    all_energies = []

    for trial in range(num_trials):
        if solver_type == 'openjij':
            sampler = oj.SQASampler()
            response = sampler.sample_qubo(
                qubo_model,
                num_reads=num_reads,
                num_sweeps=num_sweeps
            )
        elif solver_type == 'dwave_sa':
            bqm = dimod.BinaryQuadraticModel(qubo_model, 'BINARY')
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(
                bqm,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=seed + trial
            )
        else:
            raise ValueError("solver_type must be 'openjij' or 'dwave_sa'")

        for sample_data in response.data(['sample', 'energy']):
            # Convert solution to tuple for hashing
            solution_tuple = tuple(sorted(
                [idx for idx, val in sample_data.sample.items() if val == 1]
            ))
            all_solutions.append(solution_tuple)
            all_energies.append(sample_data.energy)

    # Count unique solutions
    from collections import Counter
    solution_counts = Counter(all_solutions)
    unique_solutions = len(solution_counts)

    # Find most frequent solution
    most_frequent_solution = solution_counts.most_common(1)[0]

    # Energy landscape (energy for each unique solution)
    energy_map = {}
    for sol, energy in zip(all_solutions, all_energies):
        if sol not in energy_map:
            energy_map[sol] = []
        energy_map[sol].append(energy)

    energy_landscape = {
        sol: np.mean(energies) for sol, energies in energy_map.items()
    }

    diversity_score = unique_solutions / len(all_solutions)

    print(f"  Found {unique_solutions} unique solutions out of {len(all_solutions)} samples")
    print(f"  Diversity score: {diversity_score:.4f}")
    print(f"  Most frequent solution appeared {most_frequent_solution[1]} times")

    return {
        'unique_solutions': unique_solutions,
        'total_samples': len(all_solutions),
        'solution_frequencies': solution_counts,
        'energy_landscape': energy_landscape,
        'most_frequent_solution': most_frequent_solution[0],
        'most_frequent_count': most_frequent_solution[1],
        'diversity_score': diversity_score,
        'all_energies': all_energies
    }


def convergence_analysis(qubo_model, solver_type='openjij',
                        num_sweeps_progression=None, num_reads=100, seed=42):
    """
    Analyze how solution quality improves with longer annealing times.

    Parameters:
    -----------
    qubo_model : dict
        QUBO matrix as dictionary
    solver_type : str, optional
        Either 'openjij' or 'dwave_sa'. Default: 'openjij'
    num_sweeps_progression : list, optional
        List of num_sweeps values to test. If None, uses default range.
    num_reads : int, optional
        Number of reads per sweep value. Default: 100
    seed : int, optional
        Random seed (for D-Wave SA). Default: 42

    Returns:
    --------
    pandas.DataFrame
        Results showing energy convergence with columns:
        num_sweeps, best_energy, mean_energy, std_energy, time
    """
    if num_sweeps_progression is None:
        num_sweeps_progression = [10, 50, 100, 200, 500, 1000, 2000, 5000]

    print(f"Analyzing convergence for {solver_type}...")

    results = []

    for num_sweeps in num_sweeps_progression:
        if solver_type == 'openjij':
            sampler = oj.SQASampler()
            start_time = time.time()
            response = sampler.sample_qubo(
                qubo_model,
                num_reads=num_reads,
                num_sweeps=num_sweeps
            )
            elapsed_time = time.time() - start_time

        elif solver_type == 'dwave_sa':
            bqm = dimod.BinaryQuadraticModel(qubo_model, 'BINARY')
            sampler = SimulatedAnnealingSampler()
            start_time = time.time()
            response = sampler.sample(
                bqm,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=seed
            )
            elapsed_time = time.time() - start_time
        else:
            raise ValueError("solver_type must be 'openjij' or 'dwave_sa'")

        energies = [sample.energy for sample in response.data(['energy'])]

        results.append({
            'num_sweeps': num_sweeps,
            'best_energy': np.min(energies),
            'mean_energy': np.mean(energies),
            'median_energy': np.median(energies),
            'std_energy': np.std(energies),
            'q90_energy': np.percentile(energies, 90),
            'time': elapsed_time
        })

    df = pd.DataFrame(results)
    print("Convergence analysis complete!")
    return df


if __name__ == "__main__":
    print("Annealing Benchmark Module")
    print("This module provides benchmarking tools for QUBO annealing solvers.")
    print("\nKey functions:")
    print("  - benchmark_openjij_sweep()")
    print("  - benchmark_dwave_sa_sweep()")
    print("  - compare_solvers()")
    print("  - analyze_solution_consistency()")
    print("  - convergence_analysis()")
    print("\nImport this module to use these benchmarking tools.")
