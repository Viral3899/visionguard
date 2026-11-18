@echo off
cd /d "%~dp0"
echo ============================================
echo VisionGuard AI Surveillance - Starting
echo ============================================
echo.

if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found.
  echo Run install_windows.bat first.
  pause
  exit /b 1
)

echo Checking CUDA availability...
.venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo Starting VisionGuard server...
echo Dashboard: http://127.0.0.1:8000
echo API Docs:  http://127.0.0.1:8000/docs (if enabled)
echo.
echo Press Ctrl+C to stop

.venv\Scripts\python.exe -m app.main
pause