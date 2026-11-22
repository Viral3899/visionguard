@echo off
cd /d "%~dp0"
call install_windows.bat
if errorlevel 1 (pause & exit /b 1)
call run.bat
