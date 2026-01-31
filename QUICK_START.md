# 🎯 Quick Start Guide - Skin Detection System

## Easy Setup & Run (First Time Only)

### **Option 1: Windows Batch (Easiest)**
Double-click: `run_app.bat`
- This will automatically create the virtual environment and install all dependencies

### **Option 2: PowerShell**
```powershell
.\run_app.ps1
```

### **Option 3: Manual Command Line**
```bash
cd simple_detection_system
python -m streamlit run app.py
```

---

## ✅ What's Saved?

- **`requirements-frozen.txt`** - All 60+ dependencies locked to exact versions
- **`.venv_312`** - Python 3.12 virtual environment (will be created automatically on first run)
- **`run_app.bat`** - Windows batch script for easy launching
- **`run_app.ps1`** - PowerShell script for easy launching

---

## 🚀 For Future Use

Simply run one of the scripts again. It will:
1. ✅ Check if venv exists
2. ✅ Activate the environment
3. ✅ Start the app at `http://localhost:8502`

**No reinstalling needed!**

---

## 📦 Installed Packages (60+)

- TensorFlow 2.20.0
- Keras 3.13.2
- PyTorch 2.10.0
- OpenCV 4.13.0
- YOLO (ultralytics)
- Streamlit 1.53.1
- And 50+ more dependencies...

---

## ⚠️ Note

- Requires **Python 3.12** (already installed on your system)
- Uses Python 3.14 is NOT compatible with these packages
- Total venv size: ~5-6GB (includes all ML libraries)
