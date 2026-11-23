@echo off
setlocal
cd /d "%~dp0"
echo Registers VisionGuard to start at your NEXT Windows logon.
echo Requires access to local account and working Python 3.12 virtual environment.
if not exist .venv\Scripts\python.exe (echo Install first & pause & exit /b 1)
choice /C YN /M "Create VisionGuard task in your Windows Task Scheduler"
if errorlevel 2 exit /b 1
set "TASK_FILE=%~dp0INFERENCE_MULTI_CAMERA.bat"
schtasks /Create /TN "VisionGuard-MultiCamera" /TR "\"%TASK_FILE%\"" /SC ONLOGON /F
if errorlevel 1 (
 echo Failed to register. Run from administrator terminal or use Task Scheduler GUI.
 pause
 exit /b 1
)
echo Task registered. Check the task settings to allow restart after failures.
pause
