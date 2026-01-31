@echo off
REM Simple Setup Script for Windows

echo 🎯 Simple Detection System Setup
echo =================================
echo.

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt -q

REM Create folders
echo 📁 Creating folders...
if not exist "weights" mkdir weights
if not exist "results" mkdir results

echo.
echo ✅ Setup complete!
echo.
echo 📝 Next steps:
echo    1. Add your models to weights\ folder
echo    2. Run: python detect.py
echo.
pause
