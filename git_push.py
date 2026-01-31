import subprocess
import os

os.chdir(r"c:\Users\nikhi\Downloads\simple_detection_system")

print("Step 1: Checking git status...")
result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

print("\nStep 2: Adding files...")
result = subprocess.run(["git", "add", "."], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

print("\nStep 3: Committing...")
result = subprocess.run(["git", "commit", "-m", "Fix: Wrap optional imports in try/except for Streamlit Cloud"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

print("\nStep 4: Pushing to GitHub...")
result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

print("\nDone!")
