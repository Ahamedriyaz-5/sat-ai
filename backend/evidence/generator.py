from typing import Any, Dict, Mapping
import base64

import cv2
import numpy as np

def generate_evidence_overlay(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder function for evidence overlay generation.
    In later phases, this will generate heatmaps, segmentation overlays,
    or bounding box coordinate projections.
    """
    return {"status": "success", "overlay_generated": False}


def render_mask_overlay(image: np.ndarray, masks: Mapping[str, np.ndarray]) -> str:
    """Render named binary masks over a BGR image and return a base64 PNG."""
    colors = {"water": (255, 0, 0), "built_up": (0, 0, 255), "change": (0, 255, 255)}
    overlay = image.copy()
    for label, mask in masks.items():
        overlay[mask > 0] = colors.get(label, (0, 255, 0))
    rendered = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise ValueError("Could not encode evidence overlay.")
    return base64.b64encode(encoded).decode("utf-8")
