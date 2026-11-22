@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run ONE_CLICK_SETUP_TRAIN_DEPLOY.bat or SETUP_MODELS_ONLY.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe inference.py --mode cameras
if errorlevel 1 (echo Inference exited with an error & pause & exit /b 1)
