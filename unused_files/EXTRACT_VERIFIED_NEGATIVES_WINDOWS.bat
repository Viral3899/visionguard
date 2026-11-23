@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run ONE_CLICK_SETUP_AND_RUN.bat first.
 pause
 exit /b 1
)
if "%~1"=="" (
 echo Drag a CLEAN normal CCTV video without weapons onto this .bat.
 echo Review the ENTIRE selected segment first; the script cannot verify labels.
 pause
 exit /b 2
)
echo.
echo DO NOT use video frames with bounding boxes, false-positive overlays, or real weapons.
choice /C YN /M "Did you manually inspect the video and confirm there are no guns, knives, or grenades?"
if errorlevel 2 exit /b 2
".venv\Scripts\python.exe" collect_verified_negatives.py --input "%~1" --start 0 --end 30 --interval 2 --confirm-no-weapons
if errorlevel 1 (pause & exit /b 1)
echo.
echo Review every extracted image in data\verified_negatives before training.
pause
