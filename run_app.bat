@echo off
REM Simple Skin Detection System - Run Script
REM This script automatically sets up the environment and runs the app

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv_312" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv .venv_312
    call .venv_312\Scripts\activate.bat
    echo Installing dependencies... (This may take a few minutes)
    pip install -r simple_detection_system/requirements-frozen.txt
) else (
    call .venv_312\Scripts\activate.bat
)

REM Run the Streamlit app
echo.
echo Starting Skin Detection App...
echo Open your browser and go to: http://localhost:8502
echo.
cd simple_detection_system
streamlit run app.py
pause
