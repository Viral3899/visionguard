@echo off
setlocal
cd /d "%~dp0"
call install_windows.bat
if errorlevel 1 (pause & exit /b 1)
.venv\Scripts\python.exe verify_runtime.py --require-gpu
if errorlevel 1 (
 echo GPU training requires a verified NVIDIA CUDA installation.
 echo You may run train_models.py --allow-cpu for a slower CPU training run.
 pause
 exit /b 1
)
echo.
echo CHOOSE WEAPON DATASET:
echo [1] Synthetic CCTV sample (114 images). Functional demo, NOT production training.
echo [2] Both annotated sources: CCTV + Shravya knife/pistol VOC subsets.
echo      This downloads more data and may need several GB disk space.
echo      Check the source dataset licenses before use.
choice /C 12 /N /M "Select [1/2]: "
if errorlevel 2 (
 set "DATASET_SOURCE=both"
) else (
 set "DATASET_SOURCE=syzenzz"
)
echo.
echo Starting real YOLO fine-tuning with dataset: %DATASET_SOURCE%
echo It trains only classes that actually have bounding box annotations.
set "NEGATIVE_FLAG="
if exist "data\verified_negatives" (
 echo.
 echo Found saved negative CCTV images. Include ONLY if all images contain no target weapons.
 choice /C YN /M "Have you verified every negative image?"
 if not errorlevel 2 set "NEGATIVE_FLAG=--verified-negatives data\verified_negatives"
)
.venv\Scripts\python.exe train_models.py --source %DATASET_SOURCE% --epochs 35 --base yolov8n %NEGATIVE_FLAG%
if errorlevel 1 (
 echo Training failed or data validation blocked promotion. Pretrained models were left untouched.
 pause
 exit /b 1
)
echo.
echo Newly trained weight file is now available for use on next server start.
call run.bat
