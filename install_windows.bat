@echo off
cd /d "%~dp0"
echo ============================================
echo VisionGuard AI Surveillance - Installation
echo ============================================
echo.

REM Remove old venv_cuda if exists (we use .venv now)
if exist venv_cuda (
  echo Removing old venv_cuda...
  rmdir /s /q venv_cuda
)

where py >nul 2>nul
if %errorlevel% equ 0 (
  echo Using 'py' launcher...
  py -3.11 -m venv .venv
) else (
  echo Using 'python' command...
  python -m venv .venv
)
if errorlevel 1 (
  echo.
  echo ERROR: Python venv creation failed.
  echo Make sure Python 3.11+ is installed and in PATH.
  pause & exit /b 1
)

echo.
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (echo ERROR: pip upgrade failed & pause & exit /b 1)

echo.
echo Installing PyTorch with CUDA 12.4 support (for RTX 3050/40 series)...
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo WARNING: CUDA PyTorch install failed, trying CPU version...
  .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo ERROR: Dependency installation failed & pause & exit /b 1)

echo.
echo Setting up directories...
if not exist .env copy .env.example .env
if not exist data mkdir data
if not exist data\snapshots mkdir data\snapshots
if not exist models mkdir models

echo.
echo Downloading base YOLO models...
.venv\Scripts\python.exe download_models.py

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo Next steps:
echo 1. Edit .env and set ADMIN_PASSWORD and SESSION_SECRET to strong unique values
echo 2. Run: run.bat and sign in with username admin and your ADMIN_PASSWORD
echo 3. Create client logins from Client workspaces and assign cameras to each client
echo 4. For custom fire/violence models: Place .pt files in models/ folder
echo.
pause