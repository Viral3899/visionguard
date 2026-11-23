@echo off
setlocal
cd /d "%~dp0"
call install_windows.bat
if errorlevel 1 (pause & exit /b 1)
echo Training is OPTIONAL. These models are pretrained; datasets are not downloaded.
call INFERENCE_MULTI_CAMERA.bat
