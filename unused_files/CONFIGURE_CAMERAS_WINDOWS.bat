@echo off
setlocal
cd /d "%~dp0"
if not exist cameras.json copy cameras.example.json cameras.json >nul
echo Edit camera names and RTSP sources in cameras.json.
echo For credential safety, put secret URLs into .env and use ${CAMERA_NAME} references.
notepad cameras.json
.venv\Scripts\python.exe import_cameras.py --config cameras.json --update-existing
if errorlevel 1 (echo Camera configuration failed. Check JSON/RTSP env variables. & pause & exit /b 1)
echo Camera config imported. Restart INFERENCE_MULTI_CAMERA.bat to apply.
pause
