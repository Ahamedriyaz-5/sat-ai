# RSVQA Augmented CPU Run

## Dataset

- Source: `dmarsili/RSVQA-LR-2k`, cached Hugging Face validation split
- Extracted subset: 80 real satellite images and 80 question-answer records
- Local manifest: `datasets/rsvqa_cpu80/train.jsonl`
- Existing 50-image dataset and 10-row holdout were preserved

## Training

- Base model: `Salesforce/blip-vqa-base`
- Adaptation: LoRA, rank 8, query/value target modules
- Device: CPU (`torch 2.13.0+cpu`; no CUDA device available)
- Epochs: 1
- Batch size: 1
- Learning rate: `5e-5`
- Validation split: 20% (16 records)
- Checkpoint: `checkpoints/rsvqa-blip-lora-cpu80-1e`

## Result

- Validation accuracy: `0.1875` (`3/16` exact normalized matches)
- Average training loss: `1.7726`

This adapter is retained as an experiment artifact and is not promoted over the existing adapter, which has the previously reported `0.50` (`5/10`) local holdout score. The result is not evidence that the augmented adapter is better. A meaningful comparison requires a fixed, leakage-free holdout and additional training on GPU or cloud hardware.
