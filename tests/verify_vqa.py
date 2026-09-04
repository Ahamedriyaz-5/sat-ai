import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add current workspace to path so we can import backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from fastapi.testclient import TestClient
    from backend.api.main import app
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

def create_dummy_image(color: str, size=(256, 256)) -> str:
    """Create a temporary image with a dominant color."""
    if color == "green":
        # Green pixels
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 1] = 200  # Set G channel to high
    elif color == "blue":
        # Blue pixels
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 2] = 220  # Set B channel to high
    else:
        # Gray pixels (urban simulation)
        arr = np.ones((size[1], size[0], 3), dtype=np.uint8) * 120

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"test_{color}.png")
    Image.fromarray(arr).save(file_path)
    return file_path

def test_vqa_endpoints():
    if not HAS_TESTCLIENT:
        print("Error: fastapi.testclient.TestClient could not be imported.")
        print("Please make sure fastapi and httpx (or standard dependencies) are installed.")
        sys.exit(1)

    client = TestClient(app)

    # 1. Test root endpoint
    print("Testing Root Endpoint...")
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("Root Endpoint Response:", response.json())
    print("------------------------------------------")

    # 2. Test VQA with a vegetation (green) image
    print("Testing VQA with a Vegetation (Green) image...")
    green_image_path = create_dummy_image("green")
    try:
        with open(green_image_path, "rb") as f:
            response = client.post(
                "/api/v1/vqa",
                files={"file": (os.path.basename(green_image_path), f, "image/png")},
                data={"question": "Describe the land cover and major vegetation visible in this image."}
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        print("Vegetation VQA Response:")
        print(f"  - Question: {data['query']}")
        print(f"  - Answer: {data['answer']}")
        print(f"  - Confidence: {data['confidence']}")
        print(f"  - Execution Trace: {data['execution_trace']}")
        print(f"  - Evidence Metrics: {data['evidence'].get('spectral_metrics', 'N/A')}")
        
        fallback_active = data.get("execution_trace", {}).get("fallback_active", True)
        if fallback_active:
            assert "vegetation" in data["answer"].lower() or "forest" in data["answer"].lower()
        else:
            assert len(data["answer"].strip()) > 0
    finally:
        if os.path.exists(green_image_path):
            os.remove(green_image_path)
    print("------------------------------------------")

    # 3. Test VQA with a water (blue) image
    print("Testing VQA with a Water (Blue) image...")
    blue_image_path = create_dummy_image("blue")
    try:
        with open(blue_image_path, "rb") as f:
            response = client.post(
                "/api/v1/vqa",
                files={"file": (os.path.basename(blue_image_path), f, "image/png")},
                data={"question": "Is there a river or water body present?"}
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        print("Water VQA Response:")
        print(f"  - Question: {data['query']}")
        print(f"  - Answer: {data['answer']}")
        print(f"  - Confidence: {data['confidence']}")
        print(f"  - Execution Trace: {data['execution_trace']}")
        print(f"  - Evidence Metrics: {data['evidence'].get('spectral_metrics', 'N/A')}")
        
        fallback_active = data.get("execution_trace", {}).get("fallback_active", True)
        if fallback_active:
            assert "water" in data["answer"].lower() or "river" in data["answer"].lower() or "lake" in data["answer"].lower()
        else:
            assert len(data["answer"].strip()) > 0
    finally:
        if os.path.exists(blue_image_path):
            os.remove(blue_image_path)
    print("------------------------------------------")

    print("All VQA tests passed successfully!")

if __name__ == "__main__":
    test_vqa_endpoints()
