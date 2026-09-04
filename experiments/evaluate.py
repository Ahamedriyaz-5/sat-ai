"""Compute classification accuracy and binary mask IoU/F1 from a JSONL manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.evaluation.metrics import accuracy, binary_f1, binary_iou


def mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL with prediction/target and optional pred_mask/true_mask fields")
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    output = {"samples": len(rows)}
    labels = [row for row in rows if "prediction" in row and "target" in row]
    if labels:
        output["accuracy"] = round(accuracy([row["prediction"] for row in labels], [row["target"] for row in labels]), 4)
    masks = [row for row in rows if "pred_mask" in row and "true_mask" in row]
    if masks:
        output["mean_iou"] = round(float(np.mean([binary_iou(mask(row["pred_mask"]), mask(row["true_mask"])) for row in masks])), 4)
        output["mean_f1"] = round(float(np.mean([binary_f1(mask(row["pred_mask"]), mask(row["true_mask"])) for row in masks])), 4)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
