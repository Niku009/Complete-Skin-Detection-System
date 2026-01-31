# Navigate to project
Set-Location -Path "C:\Users\nikhi\Downloads\simple_detection_system"

# Get current git status
Write-Host "Current Git Status:"
git status

# Stage changes  
Write-Host "`nAdding changes..."
git add -A

# Show what will be committed
Write-Host "`nChanges staged:"
git status --short

# Commit
Write-Host "`nCommitting..."
git commit -m "Fix: Import optional dependencies with try/except for Cloud compatibility"

# Push
Write-Host "`nPushing to GitHub..."
git push origin main

Write-Host "`nPush complete! Streamlit Cloud should auto-rebuild from the updated code."
