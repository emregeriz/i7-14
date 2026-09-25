# -*- coding: utf-8 -*-
"""YOLO eşiği, model yolu ve barkod ön işleme.

Kamera_gui/camera_gui_v2.py'den alındı (2026-09-25); preprocess_crop_for_barcode
birebir. Model paket içindeki models/ klasöründen yüklenir.
"""
import os

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.65
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_BASENAME = "V8LAST.pt"


def resolve_yolo_model_path():
    override = os.environ.get("ODAI_YOLO_MODEL", "").strip()
    if override and os.path.isfile(override):
        return override
    path = os.path.join(MODEL_DIR, MODEL_BASENAME)
    return path if os.path.isfile(path) else None


def preprocess_crop_for_barcode(crop_bgr):
    """
    Renkli kasa kırpmasını barkod okumaya uygun gri görüntüye çevirir:
    gri tonlama, CLAHE kontrast, hafif keskinleştirme; küçük kırpmaları sınırlı ölçüde büyütür.
    """
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    if len(crop_bgr.shape) == 3 and crop_bgr.shape[2] >= 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    elif len(crop_bgr.shape) == 2:
        gray = np.ascontiguousarray(crop_bgr, dtype=np.uint8)
    else:
        return None

    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return gray

    min_edge = min(h, w)
    target_min = 360
    scale = 1.0
    if min_edge < target_min:
        scale = float(target_min) / float(min_edge)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    max_side = max(nh, nw, 1)
    if max_side > 2400:
        scale *= 2400.0 / float(max_side)
        nh, nw = int(round(h * scale)), int(round(w * scale))
    if nh != h or nw != w:
        interp = cv2.INTER_CUBIC if nh > h else cv2.INTER_AREA
        gray = cv2.resize(gray, (max(1, nw), max(1, nh)), interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

