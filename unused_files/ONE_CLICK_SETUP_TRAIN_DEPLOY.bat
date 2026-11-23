@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
echo ====================================================
echo VisionGuard Python 3.12 GPU - ALL TASKS TRAINING
echo ====================================================
echo This downloads up to several GB of labeled data and models.
echo Training may consume many GPU hours and more than 15 GB disk.
echo Fire labels: D-Fire CC BY 4.0. Weapon dataset terms: see Hugging Face.
echo Fight clips: original source license/usage terms require review.
echo External .pt model weights can contain executable pickle.
echo.
choice /C YN /M "Have you reviewed all model publishers/data licenses and accept checkpoint risks"
if errorlevel 2 (echo Cancelled. & exit /b 1)
call install_windows.bat
if errorlevel 1 (echo Setup failed & pause & exit /b 1)
choice /C YN /M "Have you ALSO reviewed fight video ORIGINAL source license terms"
if errorlevel 2 (
  echo Fight video training needs license review. Train fire and weapon only.
  .venv\Scripts\python.exe setup_train_all.py --accept-checkpoint-risk --accept-data-licenses --skip-fight
) else (
  .venv\Scripts\python.exe setup_train_all.py --accept-checkpoint-risk --accept-data-licenses --fight-license-reviewed
)
if errorlevel 1 (
  echo.
  echo One or more training stages failed. See data\training_pipeline_status.json
  echo Pretrained files are not replaced by failed training.
  choice /C YN /M "Launch inference using available pretrained / qualified trained models anyway"
  if errorlevel 2 exit /b 1
)
echo.
echo Training is separate from inference. Run INFERENCE_MULTI_CAMERA.bat at any time.
call INFERENCE_MULTI_CAMERA.bat
