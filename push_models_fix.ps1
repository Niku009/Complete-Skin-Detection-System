Set-Location -Path "C:\Users\nikhi\Downloads\simple_detection_system"

Write-Host "Staging changes..."
git add simple_detection_system/app.py

Write-Host "Committing..."
git commit -m "Fix: Handle missing model weights gracefully - show warning instead of error boxes"

Write-Host "Pushing to GitHub..."
git push origin main

Write-Host "Done! Streamlit Cloud will auto-rebuild with the fix."
