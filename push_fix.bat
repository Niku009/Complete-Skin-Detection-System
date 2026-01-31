@echo off
cd /d "c:\Users\nikhi\Downloads\simple_detection_system"
echo Checking git status...
git status
echo.
echo Adding all files...
git add .
echo.
echo Current changes to commit:
git status --short
echo.
echo Committing...
git commit -m "Fix: Wrap optional imports in try/except blocks for Streamlit Cloud compatibility"
echo.
echo Pushing to GitHub...
git push origin main
echo.
echo Done!
pause
