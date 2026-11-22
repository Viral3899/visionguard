@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run ONE_CLICK_SETUP_AND_RUN.bat first to create the Python 3.12 environment.
  pause
  exit /b 1
)
.venv\Scripts\python.exe verify_runtime.py
if errorlevel 1 (pause & exit /b 1)
echo Only run trusted third-party checkpoints.
choice /C YN /M "Proceed with Hugging Face model download"
if errorlevel 2 exit /b 1
.venv\Scripts\python.exe install_security_models.py --accept-checkpoint-risk --include-3dcnn
.venv\Scripts\python.exe verify_models.py
pause
