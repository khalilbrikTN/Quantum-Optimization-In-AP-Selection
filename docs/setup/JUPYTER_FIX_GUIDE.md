# Jupyter Notebook Environment Fix

## Problem
Your Jupyter notebook is using a different Python environment than where we installed the packages.

## Solution

### Option 1: Use the New Kernel (RECOMMENDED)
1. Open your notebook in Jupyter
2. Click on the kernel selector (top right, shows current kernel name)
3. Select **"Python (Quantum Optimization)"** from the dropdown
4. The notebook will restart with the correct environment
5. Re-run your cells

### Option 2: Install in Current Jupyter Environment
If you want to keep using your current kernel, run this in a notebook cell:

```python
import sys
print(f"Current Python: {sys.executable}")

# Install packages in the current environment
!{sys.executable} -m pip install openjij dwave-ocean-sdk
```

After installation, restart the kernel (Kernel → Restart Kernel) and try again.

### Option 3: Check and Install from Terminal
Run this command to see which environment Jupyter is using:

```bash
python -c "import sys; print(sys.executable)"
```

Then install packages in that specific environment:

```bash
python -m pip install openjij dwave-ocean-sdk
```

## Verification

Add this cell at the beginning of your notebook to verify the installation:

```python
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import openjij as oj
    print("✓ OpenJij is available")
except ImportError:
    print("✗ OpenJij is NOT available")
    print(f"Install with: {sys.executable} -m pip install openjij")

try:
    from dwave.samplers import SimulatedAnnealingSampler
    print("✓ D-Wave Ocean SDK is available")
except ImportError:
    print("✗ D-Wave Ocean SDK is NOT available")
    print(f"Install with: {sys.executable} -m pip install dwave-ocean-sdk")
```

## Quick Fix (Run in Notebook Cell)

```python
import sys
import subprocess

# Install in the current notebook environment
subprocess.check_call([sys.executable, "-m", "pip", "install", "openjij", "dwave-ocean-sdk", "-q"])

print("Packages installed! Please restart the kernel: Kernel → Restart Kernel")
```

## Why This Happens

Multiple Python installations exist on your system:
- `C:\Users\Mohamed Khalil\AppData\Local\Programs\Python\Python312\python.exe` (where we installed packages)
- `C:\Users\Mohamed Khalil\AppData\Local\Microsoft\WindowsApps\python.exe`
- `C:\Users\Mohamed Khalil\AppData\Local\Python\bin\python.exe`

Your Jupyter notebook may be using a different one than the command line.
