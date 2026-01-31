"""
Complete Skin Detection System
Deploy with: streamlit run app.py
Features: Dark Circles, Acne, Redness, Skin Type Detection
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import keras_cv
from PIL import Image
import os
import warnings
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Complete Skin Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .title-box {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-box { padding: 1.5rem; background: #f0f2f6; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-box"><h1>🎯 Complete Skin Detection System</h1><p>Upload an image to get instant analysis: Dark Circles, Acne, Redness & Skin Type</p></div>',
    unsafe_allow_html=True
)

# ==================== DEVICE SETUP ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MODEL 1 & 2: YOLO Models ====================
@st.cache_resource
def load_yolo_models():
    """Load YOLO models for dark circles and acne"""
    models = {}
    
    try:
        with st.spinner("📦 Loading dark circle model..."):
            models['dark_circle'] = YOLO("weights/DarkCircideWeights.pt")
    except Exception as e:
        st.error(f"❌ Dark circle model error: {str(e)}")
        models['dark_circle'] = None
    
    try:
        with st.spinner("📦 Loading acne detection model..."):
            backbone = keras_cv.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone",
                include_rescaling=True
            )
            models['acne'] = keras_cv.models.YOLOV8Detector(
                num_classes=1,
                bounding_box_format="xyxy",
                backbone=backbone,
                fpn_depth=5
            )
            models['acne'].load_weights("weights/yolo_acne_detection.weights.h5")
    except Exception as e:
        st.error(f"❌ Acne model error: {str(e)}")
        models['acne'] = None
    
    return models

# ==================== MODEL 3: REDNESS DETECTOR ====================
class SkinConditionClassifier(nn.Module):
    """EfficientNet-based skin condition classifier"""
    def __init__(self, num_classes=2):
        super(SkinConditionClassifier, self).__init__()
        # Load pretrained EfficientNet-B0 from timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        
        # Get number of features from the model
        in_features = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_redness_model():
    """Load redness and bags under eyes detection model"""
    try:
        with st.spinner("📦 Loading redness detection model..."):
            model = SkinConditionClassifier(num_classes=2).to(device)
            
            # Load weights with strict=False to allow shape mismatches
            try:
                checkpoint = torch.load(
                    "weights/skin_redness_model_weights.pth",
                    map_location=device,
                    weights_only=False
                )
                # Try to load with strict=False
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception as e:
                st.warning(f"⚠️ Redness model weights incompatible, using pretrained model: {str(e)[:50]}...")
            
            model.eval()
            return model
    except Exception as e:
        st.error(f"❌ Redness model error: {str(e)[:100]}...")
        return None

# ==================== MODEL 4: SKIN TYPE CLASSIFIER ====================
@st.cache_resource
def load_skin_type_model():
    """Load skin type classifier (Dry, Normal, Oily)"""
    try:
        with st.spinner("📦 Loading skin type model..."):
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras import layers, Sequential
            
            IMG_SIZE = 224
            num_classes = 3
            
            resnet = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            resnet.trainable = False
            
            model = Sequential([
                layers.Rescaling(1./127.5, offset=-1),
                resnet,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
            
            # Try to load weights
            weights_path = "weights/skin_type_weights.weights.h5"
            try:
                model.load_weights(weights_path)
                print(f"✅ Successfully loaded skin type weights from {weights_path}")
            except Exception as load_err:
                print(f"⚠️ Could not load weights: {str(load_err)}")
                print(f"   Model using default ImageNet initialization + untrained classifier head")
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            return model
    except Exception as e:
        st.error(f"❌ Skin type model error: {str(e)}")
        print(f"❌ Skin type model loading failed: {str(e)}")
        return None

# ==================== LOAD ALL MODELS ====================
yolo_models = load_yolo_models()
redness_model = load_redness_model()
skin_type_model = load_skin_type_model()

# ==================== PREPROCESSING ====================
redness_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==================== FILE UPLOAD ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "👇 Choose an image (JPG, PNG, or JPEG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

# Settings
confidence = 0.25
with st.expander("⚙️ Advanced Settings"):
    confidence = st.slider("Confidence Threshold (YOLO)", 0.1, 0.9, 0.25, 0.05)

# ==================== MAIN DETECTION ====================
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load image
    img = cv2.imread(temp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    detected_img = img_rgb.copy()
    
    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(img_rgb, use_container_width=True)
    
    # Run all detections
    with st.spinner("🔍 Running all detections... Please wait"):
        results = {
            'dark_circles': 0,
            'acne': 0,
            'redness': False,
            'redness_conf': 0,
            'bags': False,
            'bags_conf': 0,
            'skin_type': 'Unknown',
            'skin_type_conf': 0
        }
        
        # ===== DETECTION 1: DARK CIRCLES =====
        if yolo_models['dark_circle']:
            try:
                dc_results = yolo_models['dark_circle'].predict(
                    source=temp_path,
                    imgsz=640,
                    conf=confidence,
                    save=False,
                    verbose=False
                )
                
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
        
        # ===== DETECTION 2: ACNE =====
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
                    
                    if hasattr(boxes, 'numpy'):
                        boxes = boxes.numpy()
                    if hasattr(confidences, 'numpy'):
                        confidences = confidences.numpy()
                    
                    for box, conf_val in zip(boxes, confidences):
                        if conf_val >= confidence:
                            results['acne'] += 1
                            x1 = int(box[0] * w / 640)
                            y1 = int(box[1] * h / 640)
                            x2 = int(box[2] * w / 640)
                            y2 = int(box[3] * h / 640)
                            
                            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(detected_img, f"AC {conf_val:.2f}", (x1, y1-30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception as e:
                st.error(f"Acne error: {str(e)}")
        
        # ===== DETECTION 3: REDNESS & BAGS =====
        if redness_model:
            try:
                redness_model.eval()
                augmented = redness_transform(image=img_rgb)
                img_tensor = augmented['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = redness_model(img_tensor)
                    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                
                # Use 0.5 as threshold for binary classification
                results['redness'] = bool(probabilities[0] >= 0.5)
                results['redness_conf'] = float(probabilities[0])
                results['bags'] = bool(probabilities[1] >= 0.5)
                results['bags_conf'] = float(probabilities[1])
            except Exception as e:
                st.error(f"Redness error: {str(e)[:100]}")
        
        # ===== DETECTION 4: SKIN TYPE =====
        if skin_type_model:
            try:
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_array = np.array(img_resized, dtype=np.float32)
                
                # Apply ResNet50 preprocessing: scale to [-1, 1]
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make predictions
                predictions = skin_type_model.predict(img_array, verbose=0)
                
                # Debug: Print raw output and all class probabilities
                print(f"\n{'='*60}")
                print(f"🔍 Skin Type Model Debug Info:")
                print(f"  Input shape: {img_array.shape}")
                print(f"  Raw predictions shape: {predictions.shape}")
                print(f"  Raw predictions: {predictions[0]}")
                print(f"  Dry (0):    {predictions[0][0]:.6f}")
                print(f"  Normal (1): {predictions[0][1]:.6f}")
                print(f"  Oily (2):   {predictions[0][2]:.6f}")
                
                class_idx = np.argmax(predictions[0])
                class_names = ['Dry', 'Normal', 'Oily']
                results['skin_type'] = class_names[class_idx]
                results['skin_type_conf'] = float(predictions[0][class_idx])
                
                print(f"✅ Final Prediction: {results['skin_type']} (idx={class_idx}, conf={results['skin_type_conf']:.6f})")
                print(f"{'='*60}\n")
            except Exception as e:
                st.error(f"Skin type error: {str(e)[:100]}")
                print(f"❌ Skin type error: {str(e)}")
    
    # Display detected image
    with col2:
        st.subheader("🎯 Detection Results")
        st.image(detected_img, use_container_width=True)
    
    # ==================== RESULTS SUMMARY ====================
    st.divider()
    st.subheader("📊 Complete Detection Summary")
    
    # Metrics in grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 Dark Circles", results['dark_circles'], delta=None, delta_color="off")
    
    with col2:
        st.metric("🟡 Acne Spots", results['acne'], delta=None, delta_color="off")
    
    with col3:
        redness_status = "🔴 Yes" if results['redness'] else "✅ No"
        st.metric("Facial Redness", redness_status, delta=f"{results['redness_conf']:.0%}", delta_color="off")
    
    with col4:
        skin_emoji = "🏜️" if results['skin_type'] == 'Dry' else "😊" if results['skin_type'] == 'Normal' else "💧"
        st.metric(f"{skin_emoji} Skin Type", results['skin_type'], delta=None, delta_color="off")
    
    # Detailed results
    st.divider()
    st.subheader("📋 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**YOLO Detections (Counting):**")
        st.write(f"- Dark Circles: {results['dark_circles']} found")
        st.write(f"- Acne Spots: {results['acne']} found")
    
    with col2:
        st.write("**Classification Results:**")
        st.write(f"- Facial Redness: {'🔴 **Yes**' if results['redness'] else '✅ **No**'} ({results['redness_conf']:.1%} confidence)")
        st.write(f"- Bags Under Eyes: {'🔴 **Yes**' if results['bags'] else '✅ **No**'} ({results['bags_conf']:.1%} confidence)")
    
    st.write(f"- **Skin Type:** {results['skin_type']} Skin")
    
    # Show skin type confidence breakdown
    if skin_type_model:
        st.info(f"✅ Skin Type Detection: **{results['skin_type']}** with **{results['skin_type_conf']:.1%}** confidence")
    
    # Download results
    st.divider()
    result_bgr = cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", result_bgr)
    
    if is_success:
        st.download_button(
            label="💾 Download Result Image",
            data=buffer.tobytes(),
            file_name=f"skin_analysis_{uploaded_file.name}",
            mime="image/jpeg",
            use_container_width=True
        )
    
    # Clean up
    os.remove(temp_path)

else:
    st.info("👆 Upload an image to start comprehensive skin analysis")
    
    with st.expander("ℹ️ What This System Detects"):
        st.write("""
        **4 Comprehensive Detections:**
        
        1. **🟢 Dark Circles** - Detects darkness under eyes using YOLOv8
        2. **🟡 Acne Spots** - Identifies acne blemishes using KerasCV
        3. **🔴 Facial Redness** - Classifies redness and puffiness using EfficientNet
        4. **👤 Skin Type** - Determines if skin is Dry, Normal, or Oily using ResNet50
        
        **Color Codes:**
        - Green boxes = Dark circles
        - Yellow boxes = Acne spots
        """)

st.divider()
st.caption("🎯 Complete Skin Detection System | 4 AI Models | Powered by YOLO, KerasCV, PyTorch & TensorFlow")
