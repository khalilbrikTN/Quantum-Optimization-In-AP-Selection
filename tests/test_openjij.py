"""Test script to verify OpenJij installation and functionality"""
import openjij as oj

print("Testing OpenJij installation...")

try:
    # Create a simple QUBO problem
    Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}

    # Create sampler
    print("Creating SQA sampler...")
    sampler = oj.SQASampler()

    # Sample the QUBO
    print("Sampling QUBO...")
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=100)

    # Get results
    best_solution = response.first.sample
    best_energy = response.first.energy

    print("SUCCESS: OpenJij is working correctly!")
    print(f"  Best solution: {best_solution}")
    print(f"  Best energy: {best_energy}")

except AttributeError as e:
    print(f"ERROR - AttributeError: {e}")
    print("  This might indicate an OpenJij API compatibility issue")
    print("  Let's check what's available in the openjij module:")
    print("  Available attributes:", dir(oj))

except Exception as e:
    print(f"ERROR: {e}")
    print(f"  Error type: {type(e).__name__}")
