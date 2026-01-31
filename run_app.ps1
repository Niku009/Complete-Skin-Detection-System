# Simple Skin Detection System - Run Script (PowerShell)
# This script automatically sets up the environment and runs the app

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
if (-Not (Test-Path ".venv_312")) {
    Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Yellow
    py -3.12 -m venv .venv_312
    & ".\.venv_312\Scripts\Activate.ps1"
    Write-Host "Installing dependencies... (This may take a few minutes)" -ForegroundColor Yellow
    pip install -r simple_detection_system/requirements-frozen.txt
} else {
    & ".\.venv_312\Scripts\Activate.ps1"
}

# Run the Streamlit app
Write-Host ""
Write-Host "Starting Skin Detection App..." -ForegroundColor Green
Write-Host "Open your browser and go to: http://localhost:8502" -ForegroundColor Cyan
Write-Host ""
Set-Location simple_detection_system
streamlit run app.py
