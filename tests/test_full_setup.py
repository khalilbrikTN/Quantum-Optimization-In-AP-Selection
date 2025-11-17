"""Comprehensive test to verify all packages are installed and working"""

print("=" * 60)
print("Testing OpenJij and D-Wave Ocean SDK installation")
print("=" * 60)

# Test 1: Import OpenJij
print("\n[1/5] Testing OpenJij import...")
try:
    import openjij as oj
    print("  SUCCESS: OpenJij imported successfully")
except ImportError as e:
    print(f"  FAILED: {e}")
    exit(1)

# Test 2: Import D-Wave packages
print("\n[2/5] Testing D-Wave imports...")
try:
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
    print("  SUCCESS: D-Wave packages imported successfully")
except ImportError as e:
    print(f"  FAILED: {e}")
    exit(1)

# Test 3: Test OpenJij SQA sampler
print("\n[3/5] Testing OpenJij SQA sampler...")
try:
    Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
    sampler = oj.SQASampler()
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=100)
    best_solution = response.first.sample
    best_energy = response.first.energy
    print(f"  SUCCESS: OpenJij SQA working - Best energy: {best_energy}")
except Exception as e:
    print(f"  FAILED: {e}")
    exit(1)

# Test 4: Test D-Wave SA sampler
print("\n[4/5] Testing D-Wave SA sampler...")
try:
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=10, num_sweeps=100)
    best_solution = response.first.sample
    best_energy = response.first.energy
    print(f"  SUCCESS: D-Wave SA working - Best energy: {best_energy}")
except Exception as e:
    print(f"  FAILED: {e}")
    exit(1)

# Test 5: Test project imports
print("\n[5/5] Testing project imports...")
try:
    import sys
    import os

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from scripts.optimization.QUBO import solve_qubo_with_openjij, solve_qubo_with_SA
    print("  SUCCESS: Project QUBO functions imported successfully")
except Exception as e:
    print(f"  FAILED: {e}")
    exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nYour environment is ready to use OpenJij and D-Wave Ocean SDK.")
print("You can now run your quantum optimization experiments.")
