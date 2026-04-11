# Simple Skin Detection System - Run Script (PowerShell)
# This script automatically sets up the environment and runs the app

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPath = Join-Path $scriptDir ".venv_312"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$streamlitEntry = "simple_detection_system\app.py"
$frozenReq = "simple_detection_system\requirements-frozen.txt"
$defaultReq = "simple_detection_system\requirements.txt"

function Test-PortInUse {
    param([int]$Port)

    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $listener
}

# Check if virtual environment exists
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Yellow
    py -3.12 -m venv $venvPath
}

if (-Not (Test-Path $pythonExe)) {
    Write-Host "ERROR: Python 3.12 venv was not created correctly." -ForegroundColor Red
    exit 1
}

# Pick requirements file automatically
$requirementsFile = if (Test-Path $frozenReq) { $frozenReq } else { $defaultReq }

# Ensure dependencies are installed (or repaired)
$streamlitInstalled = & $pythonExe -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies... (This may take a few minutes)" -ForegroundColor Yellow
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -r $requirementsFile
}

# Run the Streamlit app
$port = 8502
while (Test-PortInUse -Port $port) {
    $port++
}

Write-Host ""
Write-Host "Starting Skin Detection App..." -ForegroundColor Green
Write-Host "Open your browser and go to: http://localhost:$port" -ForegroundColor Cyan
Write-Host ""
& $pythonExe -m streamlit run $streamlitEntry --server.address 127.0.0.1 --server.port $port
