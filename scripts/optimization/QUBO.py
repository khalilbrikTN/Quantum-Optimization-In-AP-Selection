import openjij as oj
import time
import dimod
from dwave.samplers import SimulatedAnnealingSampler


def formulate_qubo(importance_scores, redundancy_matrix, k, alpha, penalty):
    """
    QUBO formulation with adaptive parameters
    """

    global relevant_aps, Q


    print(f"Formulating enhanced QUBO for k={k} APs selection...")

    relevant_aps = [ap for ap in importance_scores.keys() if importance_scores[ap] > 0]
    n_aps = len(relevant_aps)

    # Adaptive penalty based on problem size
    adaptive_penalty = penalty * (n_aps / 100.0)  # Scale with AP count

    Q = {}

    # Enhanced linear terms with importance normalization
    max_importance = max(importance_scores.values()) if importance_scores else 1.0
    for i, ap in enumerate(relevant_aps):
        normalized_importance = importance_scores[ap] / max_importance
        Q[(i, i)] = -alpha * normalized_importance

    # Enhanced quadratic terms with redundancy threshold
    redundancy_threshold = 0.3  # Only penalize high correlations
    for i in range(n_aps):
        for j in range(i+1, n_aps):
            ap_i, ap_j = relevant_aps[i], relevant_aps[j]

            redundancy_ij = redundancy_matrix.loc[ap_i, ap_j] if ap_i in redundancy_matrix.index and ap_j in redundancy_matrix.columns else 0

            # Initialize the Q entry first
            Q[(i, j)] = 0.0

            # Only apply redundancy penalty for highly correlated APs
            if redundancy_ij > redundancy_threshold:
                Q[(i, j)] += (1 - alpha) * redundancy_ij

    # Enhanced constraint with adaptive penalty
    for i in range(n_aps):
        Q[(i, i)] += adaptive_penalty * (1 - 2*k)
        for j in range(i+1, n_aps):
            Q[(i, j)] += 2 * adaptive_penalty  # Now this key definitely exists

    offset = adaptive_penalty * k**2

    print('Done')

    return Q, relevant_aps, offset


def solve_qubo_with_openjij(qubo_model, num_reads=1000, num_sweeps=1000, ):
    """
    Solves the QUBO model using the OpenJij SimulatedQuantumAnnealing sampler.
    This is a classical simulation of the quantum process that runs on your local machine.
    """
    if oj is None:
        print("ERROR: OpenJij is not installed. Cannot solve QUBO.")
        return []

    print("\nSolving QUBO with OpenJij Simulated Quantum Annealing (SQA)...")
    try:
        start_time = time.time()
        # 1. Instantiate the OpenJij SQA sampler
        sampler = oj.SQASampler()

        # 2. Solve the QUBO. OpenJij's API is very similar to D-Wave's.
        response = sampler.sample_qubo(
            qubo_model,
            num_reads = num_reads,
            num_sweeps = num_sweeps     # Annealing steps
        )

        # 3. Process the results. The best solution is the first one in the sampleset.
        best_solution = response.first.sample
        selected_indices = [idx for idx, val in best_solution.items() if val == 1]
        end_time = time.time()
        duration = end_time - start_time

        print(f"OpenJij completed in {duration:.4f} seconds")

        return selected_indices, duration
    except Exception as e:
        print(f"An error occurred during OpenJij SQA: {e}")
        return []
    
def solve_qubo_with_SA(Q, num_reads=1000, seed=42, num_sweeps=1000):
    """
    Solve QUBO using D-Wave Simulated Annealing

    Parameters:
    Q: QUBO matrix
    num_reads: Number of annealing runs
    seed: Random seed for reproducibility

    Returns:
    selected_indices: List of indices where the value is 1
    """
    print("Solving QUBO with D-Wave Simulated Annealing...")

    try:
        start_time = time.time()
        # Create BQM (Binary Quadratic Model)
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')

        # Initialize D-Wave Simulated Annealing Sampler
        sampler = SimulatedAnnealingSampler()

        # Sample solutions
        response = sampler.sample(bqm, num_reads=num_reads, seed=seed)
        response = sampler.sample(
            bqm,
            num_reads = num_reads,
            seed = seed,
            beta_range=(0.1, 5.0),            # Temperature range
            num_sweeps = num_sweeps           # Annealing duration
        )

        # Get best solution
        best_solution = response.first.sample
        best_energy = response.first.energy

        print(f"Best energy: {best_energy}")
        print(f"Selected APs: {sum(best_solution.values())}")

        # Process the results to match openjij function output structure
        selected_indices = [idx for idx, val in best_solution.items() if val == 1]

        end_time = time.time()
        duration = end_time - start_time

        print(f"SA completed in {duration:.4f} seconds")

        return selected_indices, duration

    except Exception as e:
        print(f"An error occurred during D-Wave Simulated Annealing: {e}")
        return []
    
    