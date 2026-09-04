import base64
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.api.main import app


def _write_pair() -> tuple[str, str]:
    """Create a matching pair where a green region becomes a gray urban area."""
    t1 = np.zeros((160, 160, 3), dtype=np.uint8)
    t1[:, :, 1] = 170
    t2 = t1.copy()
    t2[40:120, 40:120, :] = 135
    directory = tempfile.gettempdir()
    path_t1 = os.path.join(directory, "satquery_phase3_t1.png")
    path_t2 = os.path.join(directory, "satquery_phase3_t2.png")
    Image.fromarray(t1).save(path_t1)
    Image.fromarray(t2).save(path_t2)
    return path_t1, path_t2


def test_change_endpoint():
    path_t1, path_t2 = _write_pair()
    try:
        with TestClient(app) as client, open(path_t1, "rb") as first, open(path_t2, "rb") as second:
            response = client.post(
                "/api/v1/change",
                files={
                    "file_t1": ("t1.png", first, "image/png"),
                    "file_t2": ("t2.png", second, "image/png"),
                },
                data={"question": "Has vegetation changed between the two dates?"},
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["change_ratio"] > 0.15
        assert "decreased" in payload["answer"].lower()
        assert len(base64.b64decode(payload["change_map_b64"])) > 0
        assert payload["evidence"]["registration"]["dimensions_match"] is True
    finally:
        for path in (path_t1, path_t2):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    test_change_endpoint()
    print("Phase 3 change-analysis verification passed.")
