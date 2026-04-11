@echo off
REM Simple Skin Detection System - Run Script
REM This script automatically sets up the environment and runs the app

cd /d "%~dp0"

set "VENV=.venv_312"
set "PYTHON_EXE=%VENV%\Scripts\python.exe"
set "REQ_FROZEN=simple_detection_system\requirements-frozen.txt"
set "REQ_DEFAULT=simple_detection_system\requirements.txt"
set "APP_ENTRY=simple_detection_system\app.py"

REM Check if virtual environment exists
if not exist "%VENV%" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv %VENV%
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python 3.12 virtual environment was not created correctly.
    pause
    exit /b 1
)

set "REQ_FILE=%REQ_DEFAULT%"
if exist "%REQ_FROZEN%" (
    set "REQ_FILE=%REQ_FROZEN%"
)

"%PYTHON_EXE%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies... (This may take a few minutes)
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
)

REM Run the Streamlit app
echo.
echo Starting Skin Detection App...
echo Open your browser and go to: http://localhost:8502
echo.
"%PYTHON_EXE%" -m streamlit run "%APP_ENTRY%" --server.address 127.0.0.1 --server.port 8502
pause
