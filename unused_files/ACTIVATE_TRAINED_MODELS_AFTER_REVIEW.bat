@echo off
setlocal
cd /d "%~dp0"
echo DO NOT ACTIVATE unless you have validated the trained models using
 echo real independently labeled CCTV test clips and measured false alarms / misses.
choice /C YN /M "Have you performed independent site validation and accept the risks of missed detections"
if errorlevel 2 exit /b 1
.venv\Scripts\python.exe activate_models.py --models fire,weapon,fight --approve-site-validation
if errorlevel 1 (echo Activation not completed; review training report gates & pause & exit /b 1)
echo Restart INFERENCE_MULTI_CAMERA.bat to load selected weights.
