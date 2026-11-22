@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run ONE_CLICK_SETUP_AND_RUN.bat to install Python 3.12 and models.
 pause
 exit /b 1
)
if "%~1"=="" (
 echo Drag a video file onto this .bat, or run:
 echo   ANALYZE_VIDEO_WINDOWS.bat "C:\videos\test.mp4"
 pause
 exit /b 2
)
".venv\Scripts\python.exe" analyze_video.py --input "%~1" --every 1
if errorlevel 1 (
 echo Video analysis failed. Check above errors and run verify_models.py.
 pause
 exit /b 1
)
echo Report, events JSONL and annotated MP4 are next to the input video.
pause
