@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run SETUP_MODELS_ONLY.bat first & pause & exit /b 1)
choice /C YN /M "Have you reviewed all repository licenses and accepted third-party checkpoint risks"
if errorlevel 2 exit /b 1
choice /C YN /M "Have you reviewed the fight videos' underlying source licenses"
if errorlevel 2 (
 .venv\Scripts\python.exe setup_train_all.py --accept-checkpoint-risk --accept-data-licenses --skip-fight
) else (
 .venv\Scripts\python.exe setup_train_all.py --accept-checkpoint-risk --accept-data-licenses --fight-license-reviewed
)
if errorlevel 1 (echo Some stages failed. Read data\training_pipeline_status.json & pause & exit /b 1)
