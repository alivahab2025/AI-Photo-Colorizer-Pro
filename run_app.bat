@echo off
title AI Photo Colorizer Pro - Launcher
color 0A

echo ============================================================
echo      AI Photo Colorizer & Restorer Pro - Launcher
echo ============================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b
)

:: 2. Check and Setup Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating .venv...
    python -m venv .venv
    echo [INFO] Installing required dependencies...
    call .\.venv\Scripts\pip install --upgrade pip
    call .\.venv\Scripts\pip install "numpy<2" "setuptools<70" "fastai==1.0.61" "torchvision<0.16.0" deoldify customtkinter opencv-python pillow requests
)

:: 3. Check Required Directory Structure
if not exist "models" mkdir models
if not exist "dummy" mkdir dummy
if not exist "result_images" mkdir result_images

:: 4. Check Offline Model Files
echo [INFO] Checking offline model files in 'models' directory...
echo.

set "MISSING_MODELS=0"

if not exist "models\colorization_deploy_v2.prototxt" set "MISSING_MODELS=1"
if not exist "models\pts_in_hull.npy" set "MISSING_MODELS=1"
if not exist "models\colorization_release_v2.caffemodel" set "MISSING_MODELS=1"
if not exist "models\ColorizeArtistic_gen.pth" set "MISSING_MODELS=1"
if not exist "models\ColorizeStable_gen.pth" set "MISSING_MODELS=1"

if %MISSING_MODELS%==1 (
    color 0E
    echo =========================================================================
    echo [WARNING] Some AI Model weights are missing in the 'models' folder!
    echo.
    echo Please manually download the model files and place them inside 'models/':
    echo.
    echo  1. colorization_deploy_v2.prototxt
    echo  2. pts_in_hull.npy
    echo  3. colorization_release_v2.caffemodel
    echo  4. ColorizeArtistic_gen.pth
    echo  5. ColorizeStable_gen.pth
    echo.
    echo Refer to README.md for download links.
    echo =========================================================================
    echo.
    echo Press any key to attempt running anyway...
    pause >nul
) else (
    echo [OK] All offline model weights found successfully!
)

:: 5. Launch the Application
echo.
echo [INFO] Starting AI Photo Colorizer Pro...
echo.
call .\.venv\Scripts\python.exe app.py

pause