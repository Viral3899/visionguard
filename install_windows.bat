@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
echo ===========================================================
echo  VisionGuard - Python 3.12 x64 - CUDA PyTorch GPU Installer
echo ===========================================================
set "PY312="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
  if not errorlevel 1 set "PY312=py -3.12"
)
if not defined PY312 (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32" >nul 2>nul
    if not errorlevel 1 set "PY312=python"
  )
)
if not defined PY312 (
  echo ERROR: Python 3.12 64-bit was not found.
  echo Install Python 3.12 x64 from https://www.python.org/downloads/
  echo Tick 'Add Python to PATH' when installing, or use the Windows py launcher.
  echo Then double-click ONE_CLICK_SETUP_AND_RUN.bat again.
  exit /b 1
)
echo Using: %PY312%
%PY312% bootstrap_env.py %*
if errorlevel 1 goto :error
if not exist data mkdir data
.venv\Scripts\python.exe init_config.py
if errorlevel 1 goto :error
if not exist models mkdir models
.venv\Scripts\python.exe -c "import pathlib; p=pathlib.Path('models'); print('Downloaded models:', [x.name for x in p.glob('*.pt')])"
echo.
echo Hugging Face community .pt checkpoints may execute serialized Python code.
echo Only accept checkpoints if you trust their publishers.
set "INSTALL_SECURITY=N"
if not exist models\yolov26-fire-smoke.pt set "INSTALL_SECURITY=Y"
if not exist models\fight-yolov8-nano.pt set "INSTALL_SECURITY=Y"
if not exist models\threat-yolov8n.pt set "INSTALL_SECURITY=Y"
if /I "%INSTALL_SECURITY%"=="Y" (
  choice /C YN /M "Trust and download 3 community YOLO checkpoints for fire, fight, threat"
  if errorlevel 2 goto :skip_models
  .venv\Scripts\python.exe install_security_models.py --accept-checkpoint-risk
  if errorlevel 1 echo WARNING: Some downloads failed. Run verify_models.py.
)
:skip_models
.venv\Scripts\python.exe verify_runtime.py
if errorlevel 1 goto :error
.venv\Scripts\python.exe verify_models.py
exit /b 0
:error
echo INSTALL FAILED. Review the error messages above.
exit /b 1
