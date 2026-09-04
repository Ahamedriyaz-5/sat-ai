# BIFOLD BigEarthNet v2.0 VQA Dataset Integration

Selected dataset: `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` on Hugging Face.

## Why it fits SatQuery AI

The dataset supplies remote-sensing visual-question-answering examples based
on paired Sentinel-1 (SAR) and Sentinel-2 (optical) acquisitions. Its schema
includes `input` (question), `output` (answer), `type`, `category`, official
`split`, and the `s1_name`/`s2_name` scene identifiers. It directly supports
domain adaptation for VQA and can later provide evaluation subsets for binary
QA, multiple-choice QA, bounding boxes, and caption-like tasks.

## Important integration constraint

The Hugging Face table shown in the dataset viewer contains scene identifiers,
not an RGB image object consumed directly by the current BLIP trainer. Before
training, obtain the corresponding BigEarthNet image patches and create an
image map from each `s2_name` to a renderable RGB image path. Keep the original
multispectral/SAR source patches for future multimodal models.

For the current BLIP baseline, map Sentinel-2 B04/B03/B02 to RGB. Do not
silently discard the remaining Sentinel-2 bands for a final scientific model;
that model should use a multispectral-capable encoder.

## Reproducible preparation

```bash
python experiments/adaptation/prepare_bigearthnet_manifest.py \
  --image-map /content/s2_rgb_image_map.json \
  --output /content/bifoldearthnet_train.jsonl \
  --split train --limit 50000 --type binary
```

The script uses Hugging Face streaming, filters using the dataset row's
official `split` column, and writes the JSONL schema required by
`train_lora.py`. Begin with binary QA and a 50k-example subset for a Colab
smoke test; retain validation/test splits for held-out evaluation only.

## Required training record

Record the exact dataset revision, filter, image-map source, model, seed,
checkpoint checksum, and held-out metrics in `Docs/training_runs/`.
