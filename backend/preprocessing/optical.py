import cv2
import numpy as np


def normalize_sar_backscatter(sar_image: np.ndarray) -> np.ndarray:
    """Denoise and robustly normalize a SAR image to an 8-bit intensity map."""
    if sar_image.ndim == 3:
        sar_image = cv2.cvtColor(sar_image, cv2.COLOR_BGR2GRAY)

    filtered = cv2.medianBlur(sar_image, 3)
    low, high = np.percentile(filtered, (2, 98))
    if high <= low:
        return np.zeros_like(filtered, dtype=np.uint8)
    normalized = (filtered.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(normalized, 0, 255).astype(np.uint8)
