"""Detection pipeline.

Each detector mutates a shared ``results`` dict and draws annotations on
``detected_img``. Detectors are wrapped in try/except so one failing model
never breaks the whole analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import models


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


def detect_dark_circles(
    model,
    image_path: str,
    annotated: np.ndarray,
    confidence: float,
    result: AnalysisResult,
) -> None:
    if model is None:
        return
    try:
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
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (90, 200, 80), 2)
                cv2.putText(
                    annotated,
                    f"DC {conf_val:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (90, 200, 80),
                    2,
                )
    except Exception as e:
        result.errors.append(f"Dark circles: {str(e)[:120]}")


def detect_acne(
    model,
    image_path: str,
    annotated: np.ndarray,
    confidence: float,
    result: AnalysisResult,
) -> None:
    if model is None:
        return
    try:
        import tensorflow as tf

        h, w = annotated.shape[:2]
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

        for box, conf_val in zip(boxes, confs):
            if conf_val < confidence:
                continue
            result.acne += 1
            x1 = int(box[0] * w / 640)
            y1 = int(box[1] * h / 640)
            x2 = int(box[2] * w / 640)
            y2 = int(box[3] * h / 640)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (232, 99, 44), 2)
            cv2.putText(
                annotated,
                f"AC {conf_val:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (232, 99, 44),
                2,
            )
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

        img_resized = cv2.resize(image_rgb, (224, 224))
        arr = np.array(img_resized, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        result.skin_type = CLASS_NAMES[idx]
        result.skin_type_conf = float(preds[0][idx])
    except Exception as e:
        result.errors.append(f"Skin type: {str(e)[:120]}")


def run_pipeline(
    image_rgb: np.ndarray,
    image_path: str,
    confidence: float,
) -> tuple[AnalysisResult, np.ndarray]:
    """Run all available models and return (result, annotated image)."""
    annotated = image_rgb.copy()
    result = AnalysisResult()

    detect_dark_circles(
        models.load_dark_circle_model(),
        image_path,
        annotated,
        confidence,
        result,
    )
    detect_acne(
        models.load_acne_model(),
        image_path,
        annotated,
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
    return result, annotated
