"""Stream BIFOLD BigEarthNet VQA rows into the local LoRA JSONL schema."""
import argparse
import json
from pathlib import Path


DATASET_ID = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-map", required=True, help="JSON mapping s2_name to a local RGB image path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "val", "test"), default="train")
    parser.add_argument("--type", dest="task_type", help="Optional dataset task type, e.g. binary or bounding box")
    parser.add_argument("--limit", type=int, default=50_000)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional training dependencies first: pip install datasets") from exc
    image_map = json.loads(Path(args.image_map).read_text())
    requested_split = "validation" if args.split == "val" else args.split
    dataset = load_dataset(DATASET_ID, split="all_data", streaming=True)
    written = skipped_no_image = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in dataset:
            if row.get("split") != requested_split:
                continue
            if args.task_type and row.get("type") != args.task_type:
                continue
            image = image_map.get(row["s2_name"])
            if not image:
                skipped_no_image += 1
                continue
            stream.write(json.dumps({
                "image": image,
                "question": row["input"],
                "answer": row["output"],
                "s1_name": row["s1_name"],
                "s2_name": row["s2_name"],
                "type": row["type"],
                "category": row["category"],
            }) + "\n")
            written += 1
            if written >= args.limit:
                break
    print(json.dumps({"dataset": DATASET_ID, "split": requested_split, "written": written, "skipped_no_image": skipped_no_image}, indent=2))


if __name__ == "__main__":
    main()
