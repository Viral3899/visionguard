@echo off
cd /d "%~dp0"
echo ========================================================
echo VisionGuard AI - Live On-Screen Camera Preview
echo ========================================================
echo.
echo Select camera source to show on screen:
echo   [1] Phone / External Camera (Index 1) - Default
echo   [0] Built-in Webcam (Index 0)
echo   [U] Custom Wi-Fi / IP URL (e.g. http://phone-ip:8080/video)
echo.
set /p choice="Enter choice [1, 0, U, or press Enter for 1]: "

if "%choice%"=="" set src=1
if "%choice%"=="1" set src=1
if "%choice%"=="0" set src=0
if /i "%choice%"=="U" (
  set /p src="Enter camera stream URL: "
)

echo Starting on-screen live preview for camera %src%...
echo (Press 'Q' or 'ESC' in the video window to close)
.venv\Scripts\python.exe preview_camera.py --source %src%
pause
