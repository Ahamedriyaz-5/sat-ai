import numpy as np

from backend.evaluation.metrics import accuracy, binary_f1, binary_iou


def test_metrics():
    target = np.array([[1, 0], [1, 0]])
    prediction = np.array([[1, 0], [0, 0]])
    assert accuracy(["water", "urban"], ["water", "water"]) == 0.5
    assert binary_iou(prediction, target) == 0.5
    assert round(binary_f1(prediction, target), 4) == 0.6667
