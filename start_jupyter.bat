@echo off
echo ========================================
echo Starting Jupyter with Virtual Environment
echo ========================================
echo.
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Environment activated!
echo.
echo Starting Jupyter Notebook...
echo.
echo Remember to select: "Python (Quantum VEnv)" as your kernel
echo.
jupyter notebook
pause
