# Virtual Environment Setup Guide

## What Was Created

A completely isolated Python virtual environment has been created for this project at:
```
venv/
```

This environment is **completely separate** from your system Python and contains only the packages needed for this project.

## Packages Installed

All packages from [requirements.txt](requirements.txt) have been installed:
- **OpenJij** (v0.11.6) - Quantum annealing library
- **D-Wave Ocean SDK** (v9.0.0) - Quantum computing framework
- **scikit-learn** - Machine learning
- **pandas**, **numpy**, **scipy** - Data processing
- **matplotlib**, **seaborn** - Visualization
- **jupyter**, **notebook** - Jupyter environment
- And all their dependencies

## How to Use This Environment

### Option 1: Use in Jupyter Notebooks (RECOMMENDED)

1. Open your notebook in Jupyter
2. Click on the **kernel selector** in the top-right corner
3. Select **"Python (Quantum VEnv)"** from the dropdown
4. The notebook will restart with the virtual environment
5. All packages will now be available!

### Option 2: Activate in Command Line

**Windows (Command Prompt):**
```bash
cd "c:\Users\Mohamed Khalil\Desktop\Quantum-Optimization-In-AP-Selection"
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
cd "c:\Users\Mohamed Khalil\Desktop\Quantum-Optimization-In-AP-Selection"
venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**
```bash
cd "c:\Users\Mohamed Khalil\Desktop\Quantum-Optimization-In-AP-Selection"
source venv/Scripts/activate
```

After activation, your prompt will show `(venv)` and you can run:
```bash
python test_full_setup.py  # Test installation
jupyter notebook           # Start Jupyter
python scripts/...         # Run any Python script
```

To deactivate:
```bash
deactivate
```

### Option 3: Run Commands Directly

Without activating, you can run commands using the full path:

```bash
# On Windows
"venv/Scripts/python.exe" your_script.py
"venv/Scripts/jupyter.exe" notebook
```

## Verification

To verify everything is working, run:

```bash
# Activate the environment first (see above)
python test_full_setup.py
```

You should see:
```
============================================================
ALL TESTS PASSED!
============================================================
```

## Adding New Packages

If you need to install additional packages:

1. Activate the virtual environment (see above)
2. Install the package:
   ```bash
   pip install package_name
   ```
3. Update requirements.txt:
   ```bash
   pip freeze > requirements.txt
   ```

## Jupyter Kernel Info

A Jupyter kernel named **"Python (Quantum VEnv)"** has been installed and linked to this virtual environment.

To see all available kernels:
```bash
jupyter kernelspec list
```

To remove the kernel (if needed):
```bash
jupyter kernelspec uninstall quantum-venv
```

## Troubleshooting

### Jupyter doesn't show the new kernel
1. Restart Jupyter Lab/Notebook completely
2. Clear browser cache
3. Check kernel is installed: `jupyter kernelspec list`

### "venv not found" error
Make sure you're in the project directory:
```bash
cd "c:\Users\Mohamed Khalil\Desktop\Quantum-Optimization-In-AP-Selection"
```

### Package not found after switching kernel
1. Verify you selected "Python (Quantum VEnv)" kernel
2. Restart the kernel: Kernel → Restart Kernel
3. Check kernel in notebook: Run `import sys; print(sys.executable)`
   - Should show: `...\\venv\\Scripts\\python.exe`

## Benefits of This Setup

✅ **Isolated**: No conflicts with system Python or other projects
✅ **Reproducible**: Same environment for everyone working on the project
✅ **Clean**: Easy to delete and recreate if something breaks
✅ **Version-locked**: All packages at specific tested versions

## Recreating the Environment

If you need to recreate the environment from scratch:

```bash
# Delete old environment
rmdir /s venv  # Windows CMD
# or
rm -rf venv    # Git Bash

# Create new environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name quantum-venv --display-name "Python (Quantum VEnv)"
```

## Next Steps

1. ✅ Open your notebook: [notebooks/01_experiments/pipeline_experiment.ipynb](notebooks/01_experiments/pipeline_experiment.ipynb)
2. ✅ Select **"Python (Quantum VEnv)"** as the kernel
3. ✅ Run your experiments!

All packages are installed and ready to use. Happy coding!
