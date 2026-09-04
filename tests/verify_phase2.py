import os
import sys
import tempfile
import numpy as np
import base64
from PIL import Image

# Add workspace directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from fastapi.testclient import TestClient
    from backend.api.main import app
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

def create_mock_scene(dominant_color: str, size=(300, 300)) -> str:
    """Creates a mock satellite scene with specific features."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    if dominant_color == "green":
        # Draw a big green patch in the center (vegetation)
        arr[50:250, 50:250, 1] = 180  # High green value
        # Add some gray spots
        arr[10:30, 10:30, :] = 120
    elif dominant_color == "blue":
        # Draw a blue diagonal channel (river)
        for i in range(size[0]):
            col_start = max(0, i - 20)
            col_end = min(size[0], i + 20)
            arr[col_start:col_end, i, 2] = 200  # High blue value
    else:
        # Gray background (urban)
        arr[:, :, :] = 130
        
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"mock_{dominant_color}.png")
    Image.fromarray(arr).save(file_path)
    return file_path

def test_phase2_features():
    if not HAS_TESTCLIENT:
        print("FastAPI TestClient not available.")
        sys.exit(1)
        
    client = TestClient(app)
    
    # 1. Test Captioning with Green Scene
    print("Testing Captioning Endpoint with Vegetation image...")
    green_img = create_mock_scene("green")
    try:
        with open(green_img, "rb") as f:
            response = client.post(
                "/api/v1/caption",
                files={"file": (os.path.basename(green_img), f, "image/png")}
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        print("Captioning Response:")
        print(f"  - Caption: {data['caption']}")
        print(f"  - Confidence: {data['confidence']}")
        print(f"  - Execution Trace: {data['execution_trace']}")
        assert "vegetation" in data["caption"].lower() or "forest" in data["caption"].lower()
    finally:
        if os.path.exists(green_img):
            os.remove(green_img)
    print("------------------------------------------")

    # 2. Test Grounding with Blue Scene (Water Query)
    print("Testing Grounding Endpoint with Water query on River image...")
    blue_img = create_mock_scene("blue")
    try:
        with open(blue_img, "rb") as f:
            response = client.post(
                "/api/v1/grounding",
                files={"file": (os.path.basename(blue_img), f, "image/png")},
                data={"query": "Locate all water bodies and rivers"}
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        print("Grounding Response (Water):")
        print(f"  - Target Detected: {data['target_detected']}")
        print(f"  - Bounding Box Count: {data['box_count']}")
        print(f"  - Bounding Boxes: {data['bounding_boxes']}")
        print(f"  - Confidence: {data['confidence']}")
        print(f"  - Has base64 image: {bool(data['annotated_image_b64'])}")
        
        # Verify base64 string is decodable
        img_data = base64.b64decode(data['annotated_image_b64'])
        assert len(img_data) > 0
        assert data['box_count'] > 0
        assert len(data['bounding_boxes']) > 0
    finally:
        if os.path.exists(blue_img):
            os.remove(blue_img)
    print("------------------------------------------")

    # 3. Test Grounding with Green Scene (Vegetation Query)
    print("Testing Grounding Endpoint with Vegetation query on Forest image...")
    green_img = create_mock_scene("green")
    try:
        with open(green_img, "rb") as f:
            response = client.post(
                "/api/v1/grounding",
                files={"file": (os.path.basename(green_img), f, "image/png")},
                data={"query": "Highlight green forest area"}
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        print("Grounding Response (Vegetation):")
        print(f"  - Target Detected: {data['target_detected']}")
        print(f"  - Bounding Box Count: {data['box_count']}")
        print(f"  - Bounding Boxes: {data['bounding_boxes']}")
        print(f"  - Confidence: {data['confidence']}")
        print(f"  - Has base64 image: {bool(data['annotated_image_b64'])}")
        assert data['box_count'] > 0
    finally:
        if os.path.exists(green_img):
            os.remove(green_img)
    print("------------------------------------------")

    print("All Phase 2 Scene Captioning and Grounding tests passed successfully!")

if __name__ == "__main__":
    test_phase2_features()
