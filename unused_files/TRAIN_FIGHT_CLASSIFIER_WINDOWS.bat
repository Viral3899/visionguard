@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo First run ONE_CLICK_SETUP_AND_RUN.bat
 pause
 exit /b 1
)
echo Organize accurately labeled videos in:
echo    data\fight_clips\normal\*.mp4
echo    data\fight_clips\fight\*.mp4
echo Minimum 5 INDEPENDENT videos per class. Do not relabel screenshots as video fights.
echo.
choice /C YN /M "Train Torch R3D-18 classifier and activate for NEXT server restart"
if errorlevel 2 exit /b 0
.venv\Scripts\python.exe verify_runtime.py --require-gpu
if errorlevel 1 (pause & exit /b 1)
.venv\Scripts\python.exe train_fight_classifier.py --epochs 10 --batch 2 --activate
if errorlevel 1 (pause & exit /b 1)
echo R3D-18 trained. Check models\fight_classifier_report.json.
echo Restart run.bat to use it. Independent validation is required.
pause
