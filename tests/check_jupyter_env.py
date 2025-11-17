"""Check Jupyter environment and installed packages"""
import sys
import subprocess

print("=" * 70)
print("Jupyter Environment Check")
print("=" * 70)

print("\n1. Python Executable:")
print(f"   {sys.executable}")

print("\n2. Python Version:")
print(f"   {sys.version}")

print("\n3. Checking if openjij is installed:")
try:
    import openjij as oj
    print("   SUCCESS - openjij is installed")
except ImportError:
    print("   FAILED - openjij is NOT installed in this environment")

print("\n4. Checking if dwave-ocean-sdk is installed:")
try:
    from dwave.samplers import SimulatedAnnealingSampler
    print("   SUCCESS - dwave-ocean-sdk is installed")
except ImportError:
    print("   FAILED - dwave-ocean-sdk is NOT installed in this environment")

print("\n5. Available Jupyter kernels:")
result = subprocess.run(["jupyter", "kernelspec", "list"],
                       capture_output=True, text=True)
print(result.stdout)

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("If packages are missing, install them in THIS environment:")
print(f"  {sys.executable} -m pip install openjij dwave-ocean-sdk")
