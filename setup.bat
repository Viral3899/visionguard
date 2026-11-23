@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==================================================================
echo   VisionGuard AI Surveillance - Automated 1-Click Setup
echo ==================================================================
echo.

:: ------------------------------------------------------------------
:: 1. Check if virtual environment already exists
:: ------------------------------------------------------------------
if exist .venv\Scripts\python.exe (
    echo [1/5] Existing virtual environment detected: .venv
    goto :venv_ready
)

:: ------------------------------------------------------------------
:: 2. Detect 64-bit Python (Python 3.12, 3.11, or 3.10)
:: ------------------------------------------------------------------
echo [1/5] Detecting Python 64-bit installation...
set "PYTHON_CMD="

where py >nul 2>nul
if errorlevel 1 goto :check_python_direct

py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    goto :python_found
)
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto :python_found
)
py -3.10 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    goto :python_found
)

:check_python_direct
where python >nul 2>nul
if errorlevel 1 goto :no_python

python -c "import sys; assert sys.version_info >= (3,10) and sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
)

:no_python
echo [ERROR] No compatible 64-bit Python 3.10, 3.11, or 3.12 found.
echo Please install Python 3.12 64-bit from https://www.python.org/downloads/
echo Ensure Add Python to PATH is checked during installation.
pause
exit /b 1

:python_found
echo   Found: !PYTHON_CMD!
echo.

:: ------------------------------------------------------------------
:: 3. Provision Virtual Environment (.venv)
:: ------------------------------------------------------------------
echo [2/5] Creating virtual environment .venv...
!PYTHON_CMD! -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo   Virtual environment created successfully.
echo.

:venv_ready
echo.

:: ------------------------------------------------------------------
:: 4. GPU / Hardware Acceleration & PyTorch Installation
:: ------------------------------------------------------------------
echo [3/5] Checking GPU / Hardware acceleration and PyTorch...
set "HAS_CUDA=0"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo   NVIDIA GPU detected via nvidia-smi.
    set "HAS_CUDA=1"
) else (
    echo   No NVIDIA GPU driver utility found. Using CPU mode.
)

.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

.venv\Scripts\python.exe -c "import torch" >nul 2>nul
if errorlevel 1 (
    if "!HAS_CUDA!"=="1" (
        echo   Installing PyTorch with CUDA 12.4 acceleration...
        .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo   Installing PyTorch CPU...
        .venv\Scripts\python.exe -m pip install torch torchvision
    )
) else (
    echo   PyTorch is already installed.
)
echo.

:: ------------------------------------------------------------------
:: 5. Install Application Dependencies & Setup Configuration
:: ------------------------------------------------------------------
echo [4/5] Installing dependencies and verifying configuration...
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Some dependencies encountered warnings during install.
)

if not exist .env (
    if exist .env.example (
        copy /Y .env.example .env >nul
        echo   Created .env configuration from template.
    )
)
if not exist data mkdir data
if not exist models mkdir models
echo.

:: ------------------------------------------------------------------
:: 6. Download and Verify AI Models
:: ------------------------------------------------------------------
echo [5/5] Downloading and verifying all AI models...
.venv\Scripts\python.exe setup_models.py
if errorlevel 1 (
    echo [ERROR] Model download or verification failed. Please check internet connection.
    pause
    exit /b 1
)
echo.

:: ------------------------------------------------------------------
:: Completion & Launch Option
:: ------------------------------------------------------------------
echo ==================================================================
echo   VisionGuard AI Setup Complete and Ready!
echo ==================================================================
echo   Dashboard URL : http://127.0.0.1:8000
echo   API Docs      : http://127.0.0.1:8000/docs
echo.
echo   To launch VisionGuard at any time, run: run.bat
echo ==================================================================
echo.

choice /C YN /T 10 /D Y /M "Start VisionGuard surveillance server now (auto-starts in 10s)?"
if not errorlevel 2 (
    call run.bat
)
