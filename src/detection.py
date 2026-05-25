"""Detection pipeline.

Each detector draws annotations on a PIL Image copy and writes results into
a shared AnalysisResult dataclass. ALL cv2 usage has been replaced with
PIL so no native system library (libGL, libGLib) is required at startup.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import models


# ---------------------------------------------------------------------------
# Shared font (PIL default -- no file needed)
# ---------------------------------------------------------------------------
def _font(size: int = 12):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    dark_circles: int = 0
    acne: int = 0
    redness: bool = False
    redness_conf: float = 0.0
    bags: bool = False
    bags_conf: float = 0.0
    skin_type: str = "Unknown"
    skin_type_conf: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dark_circles": self.dark_circles,
            "acne": self.acne,
            "redness": self.redness,
            "redness_conf": self.redness_conf,
            "bags": self.bags,
            "bags_conf": self.bags_conf,
            "skin_type": self.skin_type,
            "skin_type_conf": self.skin_type_conf,
            "errors": list(self.errors),
        }


CLASS_NAMES = ["Dry", "Normal", "Oily"]


# ---------------------------------------------------------------------------
# Drawing helpers (PIL-based, no cv2)
# ---------------------------------------------------------------------------
def _draw_box(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    label: str,
    width: int = 2,
) -> None:
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    font = _font(11)
    tx, ty = x1, max(0, y1 - 14)
    # Small background behind text for readability
    try:
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    except Exception:
        draw.text((tx, ty), label, fill=color, font=font)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def detect_dark_circles(
    model,
    image_path: str,
    pil_img: Image.Image,
    confidence: float,
    result: AnalysisResult,
) -> None:
    if model is None:
        return
    try:
        draw = ImageDraw.Draw(pil_img)
        preds = model.predict(
            source=image_path,
            imgsz=640,
            conf=confidence,
            save=False,
            verbose=False,
        )
        for box in preds[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val = float(box.conf[0])
            if conf_val >= confidence:
                result.dark_circles += 1
                _draw_box(draw, x1, y1, x2, y2, (90, 200, 80), f"DC {conf_val:.2f}")
    except Exception as e:
        result.errors.append(f"Dark circles: {str(e)[:120]}")


def detect_acne(
    model,
    image_path: str,
    pil_img: Image.Image,
    confidence: float,
    result: AnalysisResult,
) -> None:
    if model is None:
        return
    try:
        import tensorflow as tf

        w, h = pil_img.size
        img_tensor = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_jpeg(img_tensor, channels=3)
        img_tensor = tf.image.resize(img_tensor, (640, 640))
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        preds = model.predict(img_tensor, verbose=0)
        if "boxes" not in preds or len(preds["boxes"]) == 0:
            return
        boxes = preds["boxes"][0]
        confs = preds["confidence"][0]
        if hasattr(boxes, "numpy"):
            boxes = boxes.numpy()
        if hasattr(confs, "numpy"):
            confs = confs.numpy()

        draw = ImageDraw.Draw(pil_img)
        for box, conf_val in zip(boxes, confs):
            if conf_val < confidence:
                continue
            result.acne += 1
            x1 = int(box[0] * w / 640)
            y1 = int(box[1] * h / 640)
            x2 = int(box[2] * w / 640)
            y2 = int(box[3] * h / 640)
            _draw_box(draw, x1, y1, x2, y2, (232, 99, 44), f"AC {conf_val:.2f}")
    except Exception as e:
        result.errors.append(f"Acne: {str(e)[:120]}")


def detect_redness_and_bags(
    model,
    transform,
    image_rgb: np.ndarray,
    result: AnalysisResult,
) -> None:
    if model is None or transform is None:
        return
    try:
        import torch

        device = models.get_torch_device()
        augmented = transform(image=image_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        result.redness = bool(probs[0] >= 0.5)
        result.redness_conf = float(probs[0])
        result.bags = bool(probs[1] >= 0.5)
        result.bags_conf = float(probs[1])
    except Exception as e:
        result.errors.append(f"Redness: {str(e)[:120]}")


def detect_skin_type(
    model,
    image_rgb: np.ndarray,
    result: AnalysisResult,
) -> None:
    if model is None:
        return
    try:
        from .tf_compat import applications
        preprocess_input = applications.resnet50.preprocess_input

        # PIL resize instead of cv2.resize
        pil = Image.fromarray(image_rgb).resize((224, 224), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        result.skin_type = CLASS_NAMES[idx]
        result.skin_type_conf = float(preds[0][idx])
    except Exception as e:
        result.errors.append(f"Skin type: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    image_rgb: np.ndarray,
    image_path: str,
    confidence: float,
) -> tuple[AnalysisResult, np.ndarray]:
    """Run all available models. Returns (result, annotated RGB ndarray)."""
    pil_img = Image.fromarray(image_rgb)
    result = AnalysisResult()

    detect_dark_circles(
        models.load_dark_circle_model(),
        image_path,
        pil_img,
        confidence,
        result,
    )
    detect_acne(
        models.load_acne_model(),
        image_path,
        pil_img,
        confidence,
        result,
    )
    detect_redness_and_bags(
        models.load_redness_model(),
        models.get_redness_transform(),
        image_rgb,
        result,
    )
    detect_skin_type(
        models.load_skin_type_model(),
        image_rgb,
        result,
    )

    annotated = np.asarray(pil_img)
    return result, annotated
