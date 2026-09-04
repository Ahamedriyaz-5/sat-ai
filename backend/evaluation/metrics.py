from typing import Iterable

import numpy as np


def accuracy(predictions: Iterable[str], targets: Iterable[str]) -> float:
    predicted, expected = list(predictions), list(targets)
    if len(predicted) != len(expected) or not expected:
        raise ValueError("Predictions and targets must be non-empty and have equal length.")
    return sum(item == target for item, target in zip(predicted, expected)) / len(expected)


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted, expected = prediction.astype(bool), target.astype(bool)
    union = np.logical_or(predicted, expected).sum()
    return float(np.logical_and(predicted, expected).sum() / union) if union else 1.0


def binary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted, expected = prediction.astype(bool), target.astype(bool)
    true_positive = np.logical_and(predicted, expected).sum()
    denominator = predicted.sum() + expected.sum()
    return float(2 * true_positive / denominator) if denominator else 1.0
