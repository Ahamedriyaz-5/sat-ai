import base64
import os
import sys
import tempfile

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.api.main import app


def _write_coregistered_pair() -> tuple[str, str]:
    optical = np.zeros((160, 160, 3), dtype=np.uint8)
    optical[:, :, :] = (35, 110, 45)
    optical[20:80, 20:80, :] = (20, 70, 190)  # RGB blue water response
    optical[90:140, 90:140, :] = 150  # neutral built-up response

    sar = np.full((160, 160), 125, dtype=np.uint8)
    sar[20:80, 20:80] = 20  # low water backscatter
    sar[90:140, 90:140] = 230  # high structural backscatter
    directory = tempfile.gettempdir()
    optical_path = os.path.join(directory, "satquery_phase4_optical.png")
    sar_path = os.path.join(directory, "satquery_phase4_sar.png")
    Image.fromarray(optical).save(optical_path)
    Image.fromarray(sar).save(sar_path)
    return optical_path, sar_path


def test_optical_sar_endpoint():
    optical_path, sar_path = _write_coregistered_pair()
    try:
        with TestClient(app) as client, open(optical_path, "rb") as optical, open(sar_path, "rb") as sar:
            response = client.post(
                "/api/v1/optical-sar",
                files={
                    "optical_file": ("optical.png", optical, "image/png"),
                    "sar_file": ("sar.png", sar, "image/png"),
                },
                data={"query": "Identify water-covered and built-up regions."},
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["class_coverage"]["water"] > 0.1
        assert payload["class_coverage"]["built_up"] > 0.08
        assert {box["class"] for box in payload["bounding_boxes"]} == {"water", "built_up"}
        assert len(base64.b64decode(payload["overlay_b64"])) > 0
        assert payload["evidence"]["coregistration"]["pair_type"] == "optical_sar"
    finally:
        for path in (optical_path, sar_path):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    test_optical_sar_endpoint()
    print("Phase 4 optical-SAR verification passed.")
