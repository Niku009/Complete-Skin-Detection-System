"""
Complete Skin Detection System - CompleteSkinDetection Premium UI
Deploy with: streamlit run app.py
Features: Dark Circles, Acne, Redness, Skin Type Detection
UI: Premium dark theme (CompleteSkinDetection) - English + Hindi mix
BACKEND: Completely unchanged from original
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import os
import warnings
import torch
import torch.nn as nn
import subprocess
import sys

# Optional imports - try to import but don't fail if missing
try:
    import keras_cv
except ImportError:
    keras_cv = None

try:
    import timm
except ImportError:
    timm = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Resolve project paths relative to this file
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(APP_DIR, "weights")
WEIGHT_FILES = {
    'dark_circle': "DarkCircideWeights.pt",
    'acne': "yolo_acne_detection.weights.h5",
    'redness': "skin_redness_model_weights.pth",
    'skin_type': "skin_type_weights.weights.h5"
}


def weight_path(model_key):
    return os.path.join(WEIGHTS_DIR, WEIGHT_FILES[model_key])


def has_weight(model_key):
    return os.path.exists(weight_path(model_key))


# ==================== MODEL WEIGHTS AUTO-DOWNLOAD ====================
# BACKEND UNCHANGED
def ensure_model_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    required_models = list(WEIGHT_FILES.values())
    missing_files = []
    for filename in required_models:
        filepath = os.path.join(WEIGHTS_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    if not missing_files:
        return
    st.warning(f"⏳ {len(missing_files)} model weights not found...")
    st.info("""
    📥 **To use all detection features, download model weights:**

    **For Local Use:**
    1. Visit: https://drive.google.com/drive/folders/15TlaZmuvhIw2c-j-AxRIp9FDi5manbUt
    2. Download all 4 model files
    3. Place in: `simple_detection_system/weights/`
    4. Run locally: `streamlit run app.py`

    **Current Status:** Running without weights (demo mode)
    """)


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="CompleteSkinDetection - त्वचा विश्लेषण",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Run auto-download at startup — BACKEND UNCHANGED
try:
    ensure_model_weights()
except Exception as e:
    st.error("⚠️ Could not download model weights automatically. Please download manually from: https://drive.google.com/drive/folders/15TlaZmuvhIw2c-j-AxRIp9FDi5manbUt")

# ==================== UI-ONLY: PREMIUM CSS INJECTION ====================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg:    #0a0a0a; --bg2: #0d0d0d; --bg3: #111;
    --bdr:   #1a1a1a; --bdr2: #252525;
    --gold:  #c9a96e; --gold2: #dfc090; --rose: #f4a7b9;
    --white: #ffffff; --mut: #555; --mut2: #333;
    --green: #4ade80; --yel: #facc15; --ora: #fb923c;
    --red:   #f87171; --blu: #60a5fa; --pur: #a78bfa;
    --serif: 'Cormorant Garamond', Georgia, serif;
    --sans:  'DM Sans', system-ui, sans-serif;
}

/* ── global ── */
.stApp { background: var(--bg) !important; font-family: var(--sans) !important; }
.stApp > header { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1200px !important; }
h1,h2,h3,h4 { font-family: var(--serif) !important; color: var(--white) !important; font-weight: 400 !important; }
p,span,label,div { font-family: var(--sans) !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--bdr2); border-radius: 3px; }

/* ── header ── */
.dh { background: var(--bg2); border-bottom: 1px solid var(--bdr); padding: 18px 28px;
      display: flex; align-items: center; justify-content: space-between;
      margin: -1rem -1rem 0 -1rem; }
.dh-logo { display: flex; align-items: center; gap: 10px; }
.dh-icon { width: 30px; height: 30px; border-radius: 7px;
            background: linear-gradient(135deg, var(--gold), var(--rose));
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; color: var(--bg); font-weight: 900; }
.dh-name { font-family: var(--serif) !important; font-size: 20px !important;
            font-weight: 600 !important; color: var(--gold) !important; }
.dh-tag  { font-size: 10px !important; color: var(--mut) !important;
            letter-spacing: 1.8px; text-transform: uppercase; font-weight: 500 !important; }

/* ── hero ── */
.hero { text-align: center; padding: 2.5rem 1rem 2rem; }
.hero-lbl { font-size: 9px !important; font-weight: 700 !important; letter-spacing: 3px !important;
             text-transform: uppercase; color: var(--gold) !important; margin-bottom: 14px; display: block; }
.hero-h1  { font-family: var(--serif) !important; font-size: clamp(30px,5vw,54px) !important;
             font-weight: 300 !important; line-height: 1.1 !important; color: var(--white) !important; margin-bottom: 12px; }
.hero-h1 em { font-style: italic;
    background: linear-gradient(90deg, var(--gold), var(--rose), #fff8f0, var(--gold));
    background-size: 200%; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; animation: shimmer 4s linear infinite; }
.hero-sub { font-size: 14px !important; color: #4a4a4a !important; font-weight: 300 !important;
             line-height: 1.8 !important; max-width: 500px; margin: 0 auto 18px; }
.pills { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
.pill  { background: var(--bg3) !important; border: 1px solid var(--bdr) !important;
          padding: 4px 12px !important; border-radius: 999px !important;
          font-size: 10px !important; color: var(--mut) !important; font-weight: 500 !important; }

/* ── upload wrap ── */
.uwrap { background: var(--bg2); border: 1px solid var(--bdr); border-radius: 16px;
          padding: 20px 20px 8px; margin: 1.5rem 0 0.5rem; }
.uwrap-title { font-family: var(--serif) !important; font-size: 17px !important;
               color: var(--gold) !important; font-weight: 400 !important; margin-bottom: 3px; }
.uwrap-sub   { font-size: 11px !important; color: var(--mut2) !important; margin-bottom: 14px; }

/* ── file uploader override ── */
[data-testid="stFileUploader"] > div {
    background: #080808 !important; border: 2px dashed var(--bdr2) !important;
    border-radius: 12px !important; transition: border-color .2s !important; }
[data-testid="stFileUploader"] > div:hover { border-color: var(--gold) !important; }
[data-testid="stFileDropzone"] p, [data-testid="stFileUploader"] label { color: var(--mut) !important; }

/* Prevent duplicate/overlapping uploader button text in translated UI states */
[data-testid="stFileUploader"] section button {
    position: relative !important;
    color: transparent !important;
}
[data-testid="stFileUploader"] section button::after {
    content: "Upload";
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--bg) !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
}

/* ── expander ── */
[data-testid="stExpander"] { background: var(--bg2) !important; border: 1px solid var(--bdr) !important;
    border-radius: 12px !important; margin-bottom: 16px !important; }
[data-testid="stExpander"] summary { color: var(--mut) !important; font-size: 12px !important;
    font-weight: 500 !important; padding: 14px 16px !important; line-height: 1.5 !important; word-break: break-word !important; }
[data-testid="stExpander"] summary svg { display: none !important; }
[data-testid="stExpander"] summary::before {
    content: "👾";
    margin-right: 8px;
    font-size: 14px;
    vertical-align: middle;
}
[data-testid="stExpander"] summary:hover { color: var(--gold) !important; }
[data-testid="stExpander"] > div > div { background: transparent !important; padding: 0 16px 14px !important; }

/* ── slider ── */
[data-testid="stSlider"] > div > div > div > div { background: var(--gold) !important; }
[data-testid="stSlider"] label { color: var(--mut) !important; font-size: 12px !important; }

/* ── spinner ── */
[data-testid="stSpinner"] p { color: var(--gold) !important; }

/* ── hr ── */
hr { border-color: var(--bdr) !important; margin: 1.2rem 0 !important; }

/* ── image border ── */
[data-testid="stImage"] img { border-radius: 14px !important; border: 1px solid var(--bdr) !important; }

/* ── section titles ── */
.sec-title { font-family: var(--serif) !important; font-size: 20px !important;
              font-weight: 400 !important; color: var(--white) !important; margin-bottom: 3px; }
.sec-sub   { font-size: 11px !important; color: var(--mut2) !important; margin-bottom: 14px; }
.img-lbl   { font-size: 9px !important; font-weight: 700 !important; letter-spacing: 2px !important;
              text-transform: uppercase !important; color: var(--mut2) !important; margin-bottom: 6px !important; }

/* ── metric grid ── */
.mgrid { display: grid; grid-template-columns: repeat(4,1fr); gap: 11px; margin: 1rem 0 1.5rem; }
@media(max-width:768px){ .mgrid { grid-template-columns: repeat(2,1fr); } }
.mcard { background: var(--bg2); border: 1px solid var(--bdr); border-radius: 14px;
          padding: 18px 14px; text-align: center; }
.mcard-icon  { font-size: 26px; margin-bottom: 9px; display: block; }
.mcard-lbl   { font-size: 8px !important; font-weight: 700 !important; letter-spacing: 2px !important;
               text-transform: uppercase !important; color: var(--mut2) !important; margin-bottom: 6px; display: block; }
.mcard-val   { font-family: var(--serif) !important; font-size: 30px !important; font-weight: 600 !important;
               color: var(--white) !important; line-height: 1 !important; display: block; margin-bottom: 8px; }
.bdg         { display: inline-block; padding: 3px 9px; border-radius: 999px;
               font-size: 10px; font-weight: 600; }
.b-g  { background: #0a2016; color: var(--green); }
.b-y  { background: #231800; color: var(--yel);   }
.b-o  { background: #1e0b00; color: var(--ora);   }
.b-r  { background: #180000; color: var(--red);   }
.b-bl { background: #0a1628; color: var(--blu);   }
.b-pu { background: #130f26; color: var(--pur);   }
.b-au { background: #1e1400; color: var(--gold);  }

/* ── detail card ── */
.dcard { background: var(--bg2); border: 1px solid var(--bdr); border-radius: 14px; padding: 18px; margin-bottom: 12px; }
.dcard-h { font-family: var(--serif) !important; font-size: 16px !important; color: var(--gold) !important;
            margin-bottom: 12px !important; font-weight: 400 !important; }
.drow  { display: flex; justify-content: space-between; align-items: center;
          padding: 8px 0; border-bottom: 1px solid #0e0e0e; }
.drow:last-child { border-bottom: none; }
.dkey  { display: flex; align-items: center; gap: 7px; font-size: 12px !important; color: #555 !important; }
.ddot  { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; display: inline-block; }
.dval  { font-size: 12px !important; font-weight: 600 !important; color: var(--white) !important; }

/* ── confidence bar ── */
.cbar-w  { display: flex; align-items: center; gap: 7px; font-size: 10px;
            color: var(--mut2) !important; margin: -2px 0 8px 12px; }
.cbar-bg { flex: 1; height: 3px; background: var(--bdr); border-radius: 2px; overflow: hidden; max-width: 180px; }
.cbar-f  { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--gold), var(--rose)); }

/* ── Streamlit metric override ── */
[data-testid="stMetric"] { background: var(--bg2) !important; border: 1px solid var(--bdr) !important;
    border-radius: 14px !important; padding: 18px 14px !important; text-align: center !important; }
[data-testid="stMetricLabel"] { color: var(--mut2) !important; font-size: 9px !important;
    text-transform: uppercase !important; letter-spacing: 1.5px !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { color: var(--white) !important; font-size: 24px !important;
    font-weight: 600 !important; font-family: var(--serif) !important; }
[data-testid="stMetricDelta"] { display: none !important; }

/* ── alerts ── */
[data-testid="stAlert"] { background: var(--bg2) !important; border: 1px solid var(--bdr) !important;
    border-radius: 10px !important; }

/* ── download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: var(--bg) !important; border: none !important; border-radius: 10px !important;
    font-family: var(--sans) !important; font-weight: 700 !important; font-size: 15px !important;
    padding: 14px 24px !important; width: 100% !important;
    transition: transform .15s, box-shadow .15s !important; }
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(201,169,110,.3) !important; }

/* ── info box ── */
.info-box { background: rgba(201,169,110,.07); border: 1px solid rgba(201,169,110,.2);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0; font-size: 13px; color: #aaa !important; }
.info-box strong { color: var(--gold) !important; }

/* ── empty state ── */
.empty { text-align: center; padding: 48px 20px; background: var(--bg2);
          border: 1px dashed var(--bdr2); border-radius: 18px; margin: 1.5rem 0; }
.empty-icon  { font-size: 50px; margin-bottom: 14px; display: block; }
.empty-title { font-family: var(--serif) !important; font-size: 26px !important;
               font-weight: 300 !important; color: var(--white) !important; margin-bottom: 8px; }
.empty-sub   { font-size: 13px !important; color: var(--mut2) !important;
               line-height: 1.75 !important; max-width: 420px; margin: 0 auto; }

/* ── footer ── */
.dfooter { text-align: center; padding: 20px 12px 12px;
            font-size: 11px !important; color: var(--mut2) !important;
            border-top: 1px solid var(--bdr); margin-top: 2rem; line-height: 1.6 !important;
            word-break: break-word !important; white-space: normal !important; }
.dfooter span { color: var(--gold) !important; }

/* ── columns gap ── */
[data-testid="stHorizontalBlock"] { gap: 14px !important; }

/* ── animations ── */
@keyframes shimmer { 0%{ background-position:0% center } 100%{ background-position:200% center } }
@keyframes fu { from{ opacity:0; transform:translateY(14px) } to{ opacity:1; transform:translateY(0) } }
.fi { animation: fu .5s ease both; }
</style>
""", unsafe_allow_html=True)

# ==================== UI-ONLY: HEADER ====================
st.markdown("""
<div class="dh">
    <div class="dh-logo">
        <div class="dh-icon">✦</div>
        <span class="dh-name">CompleteSkinDetection</span>
    </div>
    <span class="dh-tag">India's Premium Skin Analysis &nbsp;·&nbsp; त्वचा विश्लेषण</span>
</div>
""", unsafe_allow_html=True)

# ==================== UI-ONLY: HERO ====================
st.markdown("""
<div class="hero fi">
    <span class="hero-lbl">AI-Powered Dermatology · 4 CV Models</span>
    <div class="hero-h1">आपकी त्वचा deserves<br/><em>expert-level care</em></div>
    <div class="hero-sub">
        4 AI models एक ही scan में — Dark Circles, Acne, Redness &amp; Skin Type.<br/>
        अपनी photo upload करें और instant analysis पाएं।
    </div>
    <div class="pills">
        <span class="pill">👁️ Dark Circles</span>
        <span class="pill">🔴 Acne Spots</span>
        <span class="pill">🌹 Redness</span>
        <span class="pill">👜 Eye Bags</span>
        <span class="pill">💧 Skin Type</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== DEVICE SETUP — BACKEND UNCHANGED ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MODEL 1 & 2: YOLO Models — BACKEND UNCHANGED ====================
@st.cache_resource
def load_yolo_models():
    models = {}
    try:
        if has_weight('dark_circle'):
            models['dark_circle'] = YOLO(weight_path('dark_circle'))
        else:
            models['dark_circle'] = None
    except FileNotFoundError:
        models['dark_circle'] = None
    except Exception:
        models['dark_circle'] = None
    try:
        if keras_cv is not None and has_weight('acne'):
            backbone = keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone", include_rescaling=True)
            models['acne'] = keras_cv.models.YOLOV8Detector(
                num_classes=1, bounding_box_format="xyxy",
                backbone=backbone, fpn_depth=5)
            models['acne'].load_weights(weight_path('acne'))
        else:
            models['acne'] = None
    except FileNotFoundError:
        models['acne'] = None
    except Exception:
        models['acne'] = None
    return models

# ==================== MODEL 3: REDNESS DETECTOR — BACKEND UNCHANGED ====================
class SkinConditionClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(SkinConditionClassifier, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_redness_model():
    try:
        if timm is None or not has_weight('redness'):
            return None
        model = SkinConditionClassifier(num_classes=2, pretrained=False).to(device)
        checkpoint = torch.load(weight_path('redness'), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model
    except Exception:
        return None

# ==================== MODEL 4: SKIN TYPE — BACKEND UNCHANGED ====================
@st.cache_resource
def load_skin_type_model():
    try:
        if not has_weight('skin_type'):
            return None
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras import layers, Sequential
        IMG_SIZE = 224
        resnet = ResNet50(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        resnet.trainable = False
        model = Sequential([
            layers.Rescaling(1./127.5, offset=-1), resnet,
            layers.GlobalAveragePooling2D(), layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.3), layers.Dense(3, activation='softmax')])
        model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
        model.load_weights(weight_path('skin_type'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception:
        return None

# ==================== DEFERRED MODEL INIT — BACKEND UNCHANGED ====================
yolo_models = {'dark_circle': None, 'acne': None}
redness_model = None
skin_type_model = None

# ==================== PREPROCESSING — BACKEND UNCHANGED ====================
redness_transform = None
if A is not None and ToTensorV2 is not None:
    redness_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()])

# ==================== UI-ONLY: UPLOAD SECTION ====================
st.markdown("""
<div class="uwrap">
    <div class="uwrap-title">अपनी Photo Upload करें</div>
    <div class="uwrap-sub">JPG · PNG · JPEG &nbsp;|&nbsp; Good lighting में ली गई selfie best results देती है</div>
</div>
""", unsafe_allow_html=True)

# ==================== FILE UPLOAD — BACKEND CONTRACT: uploaded_file ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Image Upload",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

# ==================== ADVANCED SETTINGS — BACKEND CONTRACT: confidence ====================
confidence = 0.25
with st.expander("Advanced Settings"):
    confidence = st.slider(
        "Confidence Threshold (YOLO Detection)",
        0.1, 0.9, 0.25, 0.05,
        help="Higher = stricter detection. कम value पर ज़्यादा detections होंगी।")

# ==================== MAIN DETECTION ====================
if uploaded_file is not None:

    # Load models — BACKEND UNCHANGED
    yolo_models   = load_yolo_models()
    redness_model = load_redness_model()
    skin_type_model = load_skin_type_model()

    # Temp file save — BACKEND UNCHANGED
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load image — BACKEND UNCHANGED
    img     = cv2.imread(temp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    detected_img = img_rgb.copy()

    # ── UI-ONLY: image section header ───────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-title">📸 Image Preview</div>
    <div class="sec-sub">Original vs AI Detection overlay · आपकी तस्वीर और AI का analysis</div>
    """, unsafe_allow_html=True)

    # Original image display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="img-lbl">ORIGINAL · मूल तस्वीर</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    # ── DETECTION PIPELINE — BACKEND COMPLETELY UNCHANGED ───────────
    with st.spinner("🔍 AI Models चल रहे हैं · Analysing your skin..."):

        # BACKEND UNCHANGED
        results = {
            'dark_circles': 0, 'acne': 0,
            'redness': False, 'redness_conf': 0,
            'bags': False,    'bags_conf': 0,
            'skin_type': 'Unknown', 'skin_type_conf': 0
        }

        # ===== DETECTION 1: DARK CIRCLES — BACKEND UNCHANGED =====
        if yolo_models['dark_circle']:
            try:
                dc_results = yolo_models['dark_circle'].predict(
                    source=temp_path, imgsz=640, conf=confidence, save=False, verbose=False)
                for box in dc_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_val = float(box.conf[0])
                    if conf_val >= confidence:
                        results['dark_circles'] += 1
                        cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(detected_img, f"DC {conf_val:.2f}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Dark circle error: {str(e)}")

        # ===== DETECTION 2: ACNE — BACKEND UNCHANGED =====
        if yolo_models['acne']:
            try:
                img_tensor = tf.io.read_file(temp_path)
                img_tensor = tf.image.decode_jpeg(img_tensor, channels=3)
                img_tensor = tf.image.resize(img_tensor, (640, 640))
                img_tensor = tf.cast(img_tensor, tf.float32)
                img_tensor = tf.expand_dims(img_tensor, axis=0)
                acne_results = yolo_models['acne'].predict(img_tensor, verbose=0)
                if 'boxes' in acne_results and len(acne_results['boxes']) > 0:
                    boxes = acne_results['boxes'][0]
                    confidences = acne_results['confidence'][0]
                    if hasattr(boxes, 'numpy'):        boxes = boxes.numpy()
                    if hasattr(confidences, 'numpy'): confidences = confidences.numpy()
                    for box, conf_val in zip(boxes, confidences):
                        if conf_val >= confidence:
                            results['acne'] += 1
                            x1 = int(box[0] * w / 640); y1 = int(box[1] * h / 640)
                            x2 = int(box[2] * w / 640); y2 = int(box[3] * h / 640)
                            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(detected_img, f"AC {conf_val:.2f}", (x1, y1-30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception as e:
                st.error(f"Acne error: {str(e)}")

        # ===== DETECTION 3: REDNESS & BAGS — BACKEND UNCHANGED =====
        if redness_model and redness_transform is not None:
            try:
                redness_model.eval()
                augmented  = redness_transform(image=img_rgb)
                img_tensor = augmented['image'].unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs       = redness_model(img_tensor)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                results['redness']      = bool(probabilities[0] >= 0.5)
                results['redness_conf'] = float(probabilities[0])
                results['bags']         = bool(probabilities[1] >= 0.5)
                results['bags_conf']    = float(probabilities[1])
            except Exception as e:
                st.error(f"Redness error: {str(e)[:100]}")

        # ===== DETECTION 4: SKIN TYPE — BACKEND UNCHANGED =====
        if skin_type_model:
            try:
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_array   = np.array(img_resized, dtype=np.float32)
                img_array   = tf.keras.applications.resnet50.preprocess_input(img_array)
                img_array   = np.expand_dims(img_array, axis=0)
                predictions = skin_type_model.predict(img_array, verbose=0)
                print(f"\n{'='*60}")
                print(f"🔍 Skin Type Model Debug Info:")
                print(f"  Input shape: {img_array.shape}")
                print(f"  Raw predictions: {predictions[0]}")
                print(f"  Dry (0): {predictions[0][0]:.6f}  Normal (1): {predictions[0][1]:.6f}  Oily (2): {predictions[0][2]:.6f}")
                class_idx = np.argmax(predictions[0])
                class_names = ['Dry', 'Normal', 'Oily']
                results['skin_type']      = class_names[class_idx]
                results['skin_type_conf'] = float(predictions[0][class_idx])
                print(f"✅ Final: {results['skin_type']} conf={results['skin_type_conf']:.6f}")
                print(f"{'='*60}\n")
            except Exception as e:
                st.error(f"Skin type error: {str(e)[:100]}")
                print(f"❌ Skin type error: {str(e)}")

    # Detected image display
    with col2:
        st.markdown('<div class="img-lbl">AI DETECTION · विश्लेषण</div>', unsafe_allow_html=True)
        st.image(detected_img, use_container_width=True)

    # ==================== UI-ONLY: RESULTS SUMMARY ====================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-title">📊 Detection Summary · विश्लेषण सारांश</div>
    <div class="sec-sub">4 AI models के results एक नज़र में</div>
    """, unsafe_allow_html=True)

    # ── helpers ──────────────────────────────────────────────────────
    def count_badge(n):
        if n == 0:   return '<span class="bdg b-g">✓ None</span>'
        elif n <= 2: return '<span class="bdg b-y">● Mild</span>'
        elif n <= 5: return '<span class="bdg b-o">● Moderate</span>'
        else:        return '<span class="bdg b-r">● Severe</span>'

    def bool_badge(detected, conf):
        if not detected: return '<span class="bdg b-g">✓ Clear</span>'
        if conf >= 0.75: return '<span class="bdg b-r">● Severe</span>'
        if conf >= 0.50: return '<span class="bdg b-o">● Moderate</span>'
        return              '<span class="bdg b-y">● Mild</span>'

    skin_badge = {
        'Dry':     '<span class="bdg b-bl">💧 Dry</span>',
        'Normal':  '<span class="bdg b-g">😊 Normal</span>',
        'Oily':    '<span class="bdg b-y">✨ Oily</span>',
        'Unknown': '<span class="bdg b-au">— Unknown</span>',
    }

    # ── 4 metric cards ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="mgrid">
        <div class="mcard">
            <span class="mcard-icon">👁️</span>
            <span class="mcard-lbl">Dark Circles · काले घेरे</span>
            <span class="mcard-val">{results['dark_circles']}</span>
            {count_badge(results['dark_circles'])}
        </div>
        <div class="mcard">
            <span class="mcard-icon">🔴</span>
            <span class="mcard-lbl">Acne Spots · मुंहासे</span>
            <span class="mcard-val">{results['acne']}</span>
            {count_badge(results['acne'])}
        </div>
        <div class="mcard">
            <span class="mcard-icon">🌹</span>
            <span class="mcard-lbl">Facial Redness · लालिमा</span>
            <span class="mcard-val">{"Yes" if results['redness'] else "No"}</span>
            {bool_badge(results['redness'], results['redness_conf'])}
        </div>
        <div class="mcard">
            <span class="mcard-icon">💧</span>
            <span class="mcard-lbl">Skin Type · त्वचा प्रकार</span>
            <span class="mcard-val">{results['skin_type']}</span>
            {skin_badge.get(results['skin_type'], skin_badge['Unknown'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== UI-ONLY: DETAILED ANALYSIS ====================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-title">📋 Detailed Analysis · विस्तृत विश्लेषण</div>
    <div class="sec-sub">Each model का complete breakdown</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="dcard">
            <div class="dcard-h">🎯 YOLO Object Detection</div>
            <div class="drow">
                <div class="dkey"><span class="ddot" style="background:#4ade80"></span>Dark Circles · काले घेरे</div>
                <span class="dval">{results['dark_circles']} detected</span>
            </div>
            <div class="drow">
                <div class="dkey"><span class="ddot" style="background:#facc15"></span>Acne Spots · मुंहासे</div>
                <span class="dval">{results['acne']} detected</span>
            </div>
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid #0e0e0e;">
                <div style="font-size:9px;color:#2a2a2a;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">
                    Confidence Threshold
                </div>
                <div class="cbar-w">
                    <span>{confidence:.0%}</span>
                    <div class="cbar-bg"><div class="cbar-f" style="width:{confidence*100:.0f}%"></div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        r_pct  = f"{results['redness_conf']:.1%}"
        b_pct  = f"{results['bags_conf']:.1%}"
        sk_pct = f"{results['skin_type_conf']:.1%}"
        st.markdown(f"""
        <div class="dcard">
            <div class="dcard-h">🧠 Classification Models</div>
            <div class="drow">
                <div class="dkey"><span class="ddot" style="background:#f04828"></span>Facial Redness · लालिमा</div>
                <span class="dval">{"🔴 Yes" if results['redness'] else "✅ No"} &nbsp;·&nbsp; {r_pct}</span>
            </div>
            <div class="cbar-w"><div class="cbar-bg"><div class="cbar-f" style="width:{results['redness_conf']*100:.0f}%"></div></div><span>{r_pct}</span></div>
            <div class="drow">
                <div class="dkey"><span class="ddot" style="background:#2864d8"></span>Eye Bags · आंखों की सूजन</div>
                <span class="dval">{"🔴 Yes" if results['bags'] else "✅ No"} &nbsp;·&nbsp; {b_pct}</span>
            </div>
            <div class="cbar-w"><div class="cbar-bg"><div class="cbar-f" style="width:{results['bags_conf']*100:.0f}%"></div></div><span>{b_pct}</span></div>
            <div class="drow">
                <div class="dkey"><span class="ddot" style="background:#c9a96e"></span>Skin Type · त्वचा प्रकार</div>
                <span class="dval">{results['skin_type']} &nbsp;·&nbsp; {sk_pct}</span>
            </div>
            <div class="cbar-w"><div class="cbar-bg"><div class="cbar-f" style="width:{results['skin_type_conf']*100:.0f}%"></div></div><span>{sk_pct}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Skin type info box — BACKEND CHECK UNCHANGED
    if skin_type_model:
        st.markdown(f"""
        <div class="info-box">
            ✦ &nbsp; Skin Type Detected: &nbsp;
            <strong>{results['skin_type']}</strong> &nbsp;|&nbsp;
            Confidence: <strong>{results['skin_type_conf']:.1%}</strong>
        </div>
        """, unsafe_allow_html=True)

    # ==================== UI-ONLY: DOWNLOAD ====================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-title">💾 Download Results · रिपोर्ट Save करें</div>
    <div class="sec-sub">AI-annotated image अपने records के लिए download करें</div>
    """, unsafe_allow_html=True)

    # BACKEND UNCHANGED
    result_bgr = cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", result_bgr)
    if is_success:
        st.download_button(
            label="✦  Download Analysis Report · रिपोर्ट Download करें",
            data=buffer.tobytes(),
            file_name=f"completeskindetection_analysis_{uploaded_file.name}",
            mime="image/jpeg",
            use_container_width=True
        )

    # BACKEND UNCHANGED
    os.remove(temp_path)

# ==================== UI-ONLY: EMPTY STATE ====================
else:
    st.markdown("""
    <div class="empty fi">
        <span class="empty-icon">🖼️</span>
        <div class="empty-title">Upload करें और जानें</div>
        <div class="empty-sub">
            एक clear selfie upload करें — अच्छी lighting में।<br/>
            4 AI models मिलकर आपकी skin का complete analysis करेंगे।<br/><br/>
            <em>Supports JPG · PNG · JPEG</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # BACKEND UNCHANGED
    with st.expander("What Does It Detect?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="dcard">
                <div class="dcard-h">🎯 YOLO Detection</div>
                <div class="drow"><div class="dkey"><span class="ddot" style="background:#4ade80"></span>Dark Circles</div><span class="dval">YOLOv8</span></div>
                <div class="drow"><div class="dkey"><span class="ddot" style="background:#facc15"></span>Acne Spots</div><span class="dval">KerasCV</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="dcard">
                <div class="dcard-h">🧠 Classification</div>
                <div class="drow"><div class="dkey"><span class="ddot" style="background:#f04828"></span>Redness + Eye Bags</div><span class="dval">EfficientNet-B0</span></div>
                <div class="drow"><div class="dkey"><span class="ddot" style="background:#c9a96e"></span>Skin Type</div><span class="dval">ResNet50</span></div>
            </div>
            """, unsafe_allow_html=True)

# ==================== UI-ONLY: FOOTER ====================
st.markdown("""
<div class="dfooter">
    <span>✦ CompleteSkinDetection</span><br/>
    4 AI Models · YOLOv8 · KerasCV · EfficientNet · ResNet50<br/>
    <span>India's Premium Skin Analysis</span>
</div>
""", unsafe_allow_html=True)
