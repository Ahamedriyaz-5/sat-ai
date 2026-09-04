import base64
import os
import sys
import tempfile

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.agent.graph import agent_graph
from backend.agent.state import AgentState
from backend.api.main import app


def _image(path: str, rgb: tuple[int, int, int]):
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    array[:, :, :] = rgb
    Image.fromarray(array).save(path)


def test_agent_routes_and_reports():
    directory = tempfile.gettempdir()
    first, second = os.path.join(directory, "agent_one.png"), os.path.join(directory, "agent_two.png")
    _image(first, (0, 180, 0))
    _image(second, (130, 130, 130))
    try:
        with TestClient(app) as client, open(first, "rb") as file:
            response = client.post(
                "/api/v1/agent",
                files={"file_1": ("one.png", file, "image/png")},
                data={"query": "Describe this scene", "include_report": "true", "thread_id": "test_session_1"},
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["route"]["task"] == "caption"
        assert payload["thread_id"] == "test_session_1"
        assert base64.b64decode(payload["report_pdf_b64"]).startswith(b"%PDF")
        assert len(payload["execution_trace"]["steps"]) >= 2

        with TestClient(app) as client, open(first, "rb") as one, open(second, "rb") as two:
            response = client.post(
                "/api/v1/agent",
                files={"file_1": ("one.png", one, "image/png"), "file_2": ("two.png", two, "image/png")},
                data={"query": "What changed between dates?"},
            )
        assert response.status_code == 200, response.text
        assert response.json()["route"]["task"] == "change"
    finally:
        for path in (first, second):
            if os.path.exists(path):
                os.remove(path)


def test_direct_stategraph_vqa_and_grounding():
    directory = tempfile.gettempdir()
    img_path = os.path.join(directory, "agent_vqa.png")
    _image(img_path, (0, 0, 200))  # Water
    try:
        # 1. Test VQA route
        vqa_state: AgentState = {
            "query": "Is there a river or water body present?",
            "image_count": 1,
            "requested_task": "auto",
            "file_1_path": img_path,
            "thread_id": "direct_vqa_thread",
        }
        res_vqa = agent_graph.invoke(vqa_state, config={"configurable": {"thread_id": "direct_vqa_thread"}})
        out_vqa = res_vqa["final_output"]
        assert out_vqa["status"] == "success"
        assert out_vqa["route"]["task"] == "vqa"
        assert "water body is detected" in out_vqa["answer"].lower()

        # 2. Test Grounding route
        ground_state: AgentState = {
            "query": "Highlight the water body",
            "image_count": 1,
            "requested_task": "auto",
            "file_1_path": img_path,
            "thread_id": "direct_ground_thread",
        }
        res_ground = agent_graph.invoke(ground_state)
        out_ground = res_ground["final_output"]
        assert out_ground["status"] == "success"
        assert out_ground["route"]["task"] == "grounding"
        assert len(out_ground.get("bounding_boxes", [])) >= 1
        assert out_ground.get("annotated_image_b64")
        assert out_ground.get("overlay_b64") == out_ground["annotated_image_b64"]
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def test_direct_stategraph_optical_sar():
    directory = tempfile.gettempdir()
    optical = os.path.join(directory, "agent_opt.png")
    sar = os.path.join(directory, "agent_sar.png")
    _image(optical, (0, 180, 0))
    _image(sar, (150, 150, 150))
    try:
        state: AgentState = {
            "query": "Use optical and SAR to identify built-up and water features",
            "image_count": 2,
            "requested_task": "auto",
            "file_1_path": optical,
            "file_2_path": sar,
            "thread_id": "sar_thread",
        }
        result = agent_graph.invoke(state)
        output = result["final_output"]
        assert output["status"] == "success"
        assert output["route"]["task"] == "optical_sar"
        assert output.get("overlay_b64") is not None
        assert output["pair_metadata"]["pair_type"] == "optical_sar"

        # Verify state checkpoint history was recorded
        history = agent_graph.get_state_history("sar_thread")
        assert len(history) >= 1
    finally:
        for path in (optical, sar):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    test_agent_routes_and_reports()
    test_direct_stategraph_vqa_and_grounding()
    test_direct_stategraph_optical_sar()
    print("All LangGraph StateGraph agent tests passed successfully!")

