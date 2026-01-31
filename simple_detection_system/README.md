# 🎯 Complete Skin Detection System

**A comprehensive AI-powered skin analysis tool** that detects dark circles, acne, skin redness, and identifies skin types from facial images.

## ✨ Features

✅ **4 AI Models Integrated:**
- 🌙 Dark Circle Detection (YOLO)
- 🔴 Acne Detection (Keras-CV YOLO)
- 🔴 Skin Redness Analysis (PyTorch)
- 👤 Skin Type Classification (EfficientNet)

✅ **Web-Based Interface** - Upload images and get instant analysis
✅ **Real-time Results** - Bounding boxes and classification scores
✅ **Download Feature** - Save annotated results
✅ **GPU Support** - CUDA acceleration when available

---

## 📁 Project Structure

```
simple_detection_system/
├── app.py                         # Main Streamlit web app
├── requirements.txt               # All dependencies
├── requirements-deploy.txt        # Deployment packages
├── README.md                      # This file
├── GUIDE.md                       # Detailed setup guide
├── weights/                       # Model weights directory
│   ├── DarkCircideWeights.pt      # Dark circles (YOLO)
│   ├── yolo_acne_detection.weights.h5    # Acne detection
│   ├── skin_redness_model_weights.pth    # Redness analysis
│   └── skin_type_weights.weights.h5      # Skin type classifier
├── run_app.bat                    # Windows launcher (automatic setup)
├── run_app.ps1                    # PowerShell launcher
└── setup.bat / setup.sh           # Manual setup scripts
```

---

## � Download Model Weights (Required!)

**⚠️ Important:** Model weight files are NOT included in the repository due to their large size (350MB total).

### **Step 1: Download Weights from Google Drive**

👉 **[Download All Model Weights (350MB)](https://drive.google.com/drive/folders/15TlaZmuvhIw2c-j-AxRIp9FDi5manbUt?usp=sharing)**

The folder contains 4 files (total ~350MB)

### **Step 2: Place Weights in Correct Folder**

1. Download all files from the Google Drive folder
2. Navigate to your project: `simple_detection_system/`
3. Open the `weights/` folder
4. Copy all 4 model files (`.pt`, `.h5`, `.pth`) into `weights/`

**Folder structure should look like:**
```
simple_detection_system/weights/
├── DarkCircideWeights.pt
├── yolo_acne_detection.weights.h5
├── skin_redness_model_weights.pth
└── skin_type_weights.weights.h5
```

---

## �🚀 Quick Start

### **Windows (Easiest)**
```bash
double-click run_app.bat
```
The app will automatically:
- Create Python 3.12 virtual environment
- Install all 60+ dependencies
- Launch at `http://localhost:8502`

### **PowerShell**
```powershell
.\run_app.ps1
```

### **Manual Setup**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔧 System Requirements

- **Python:** 3.12 (3.10+ compatible)
- **RAM:** 4GB minimum (8GB+ recommended)
- **GPU:** Optional (CUDA support for faster processing)
- **Disk:** ~6GB (for virtual environment with all ML libraries)

---

## 📊 Supported Image Formats

- `.jpg`, `.jpeg`, `.png`
- Recommended: 1024x768 pixels or larger
- Max file size: 200MB (Streamlit default)

---

## 🎨 Analysis Output

For each uploaded image, you'll get:
- **Dark Circles:** Detected regions with bounding boxes (green)
- **Acne:** Detected blemishes with bounding boxes (yellow)
- **Skin Redness:** Intensity score (0-100%)
- **Skin Type:** Classification (Oily/Normal/Dry)
- **Downloadable Result:** Annotated image with all detections

---

## 📦 Dependencies

- TensorFlow 2.15.0
- Keras 2.15.0 + Keras-CV 0.9.0
- PyTorch 2.10.0
- OpenCV 4.10.0
- Ultralytics 8.3.0
- Streamlit 1.28.0
- Pillow, NumPy, Matplotlib

See `requirements-deploy.txt` for complete list.

---


## 📝 Usage Example

1. Open the web app
2. Upload a facial image (JPG/PNG)
3. Wait for all 4 models to process (~2-5 seconds)
4. View results with annotated image
5. Download the result image if needed

---

## 🐛 Troubleshooting

**App won't start?**
- Delete `.venv_312` folder and re-run launcher

**Model not loading?**
- Verify all weight files in `weights/` folder
- Check file names match exactly

**Slow processing?**
- GPU not detected? Install CUDA + cuDNN
- Reduce image size for faster processing

---

## 📄 License

This project includes pre-trained models. Check individual model licenses before commercial use.

---

## 🤝 Support

For issues or questions, check `GUIDE.md` for detailed setup instructions.
