@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe verify_deployment.py --probe --require-gpu --strict
if errorlevel 1 (
 echo Some readiness checks failed. Check cameras.json, RTSP, GPU and credentials.
 pause
 exit /b 1
)
echo Checks passed. Actual detections still require labeled CCTV validation.
pause
