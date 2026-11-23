@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  set /p "VIDEO=Full path to input video: "
) else (
  set "VIDEO=%~1"
)
if not exist "%VIDEO%" (echo Video not found & pause & exit /b 1)
.venv\Scripts\python.exe inference.py --mode video --input "%VIDEO%"
if errorlevel 1 (echo Inference failed & pause & exit /b 1)
