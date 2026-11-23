@echo off
cd /d "%~dp0"
echo ============================================================
echo VisionGuard AI - Phone Camera & Connected Device Launcher
echo ============================================================
echo.

if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run run.bat or install_windows.bat first.
  pause
  exit /b 1
)

.venv\Scripts\python.exe phone_camera_setup.py
echo.
pause
