import base64
import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from backend.models.base import BaseSpecialistModel
from backend.preprocessing.optical import normalize_sar_backscatter


class OpticalSARFusionModel(BaseSpecialistModel):
    """Fuse optical spectral and SAR backscatter evidence for land-cover cues.

    This lightweight baseline is designed for reproducible local execution.
    Water is supported by blue optical response plus low SAR backscatter;
    built-up areas are supported by neutral optical pixels plus high SAR
    backscatter. A learned fusion model can later replace these masks through
    the same specialist interface.
    """

    @property
    def name(self) -> str:
        return "OpticalSARFusion"

    @property
    def version(self) -> str:
        return "1.0.0"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        optical_path = inputs.get("optical_path")
        sar_path = inputs.get("sar_path")
        query = inputs.get("query", "Identify water-covered and built-up regions.").strip()
        if not optical_path or not os.path.exists(optical_path):
            raise FileNotFoundError(f"Optical image not found: {optical_path}")
        if not sar_path or not os.path.exists(sar_path):
            raise FileNotFoundError(f"SAR image not found: {sar_path}")

        start_time = time.time()
        optical = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        sar = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
        if optical is None or sar is None:
            raise ValueError("Could not read the optical or SAR image.")
        if optical.shape[:2] != sar.shape[:2]:
            raise ValueError("Optical and SAR images must have matching dimensions.")

        sar_normalized = normalize_sar_backscatter(sar)
        b, g, r = cv2.split(optical)
        b_int, g_int, r_int = b.astype(int), g.astype(int), r.astype(int)
        brightness = (b_int + g_int + r_int) / 3.0
        channel_spread = np.maximum.reduce((np.abs(r_int - g_int), np.abs(g_int - b_int), np.abs(r_int - b_int)))

        optical_water = (b_int > g_int + 15) & (b_int > r_int + 15) & (brightness < 235)
        optical_built = (channel_spread < 20) & (brightness > 80) & (brightness < 220)
        sar_low = sar_normalized < np.percentile(sar_normalized, 35)
        sar_high = sar_normalized > np.percentile(sar_normalized, 65)

        water_mask = (optical_water & sar_low).astype(np.uint8) * 255
        builtup_mask = (optical_built & sar_high).astype(np.uint8) * 255
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        builtup_mask = cv2.morphologyEx(builtup_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        overlay = optical.copy()
        overlay[water_mask > 0] = (255, 0, 0)  # blue in BGR
        overlay[builtup_mask > 0] = (0, 0, 255)  # red in BGR
        rendered = cv2.addWeighted(optical, 0.55, overlay, 0.45, 0)
        boxes = self._boxes(water_mask, "water") + self._boxes(builtup_mask, "built_up")
        _, buffer = cv2.imencode(".png", rendered)

        total_pixels = optical.shape[0] * optical.shape[1]
        water_ratio = float(np.count_nonzero(water_mask) / total_pixels)
        builtup_ratio = float(np.count_nonzero(builtup_mask) / total_pixels)
        confidence = float(np.clip(0.65 + min(water_ratio + builtup_ratio, 0.5) * 0.5, 0.65, 0.9))
        return {
            "summary": (
                f"Optical-SAR fusion found water across {water_ratio * 100:.1f}% and "
                f"built-up candidates across {builtup_ratio * 100:.1f}% of the scene."
            ),
            "query": query,
            "class_coverage": {"water": round(water_ratio, 4), "built_up": round(builtup_ratio, 4)},
            "bounding_boxes": boxes,
            "overlay_b64": base64.b64encode(buffer).decode("utf-8"),
            "confidence": round(confidence, 4),
            "evidence": {
                "fusion_rule": "water = optical blue response ∩ low SAR backscatter; built_up = neutral optical response ∩ high SAR backscatter",
                "sar_preprocessing": "3x3 median filter and 2nd/98th percentile normalization",
            },
            "execution_trace": {
                "task": "Cross-modal Optical + SAR Fusion",
                "model": f"{self.name} (Spectral-Backscatter Baseline)",
                "execution_time_seconds": round(time.time() - start_time, 4),
            },
        }

    @staticmethod
    def _boxes(mask: np.ndarray, class_name: str) -> List[Dict[str, Any]]:
        minimum_area = max(50, int(mask.size * 0.0005))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append({"class": class_name, "coordinates": [y, x, y + height, x + width]})
        return boxes
