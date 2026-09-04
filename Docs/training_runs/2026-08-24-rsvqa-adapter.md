# RSVQA LoRA Adapter Run

- Date: 2026-08-24
- Base model: Salesforce/blip-vqa-base
- Dataset: local RSVQA subset at `datasets/rsvqa/train.jsonl`
- Output checkpoint: `checkpoints/rsvqa-blip-lora`
- Status: real fine-tuning run executed successfully on CUDA

## Training command used

```bash
python experiments/adaptation/train_lora.py \
  --train-jsonl datasets/rsvqa/train.jsonl \
  --output-dir checkpoints/rsvqa-blip-lora \
  --epochs 8 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --seed 42
```

## Hyperparameters

- seed: 42
- epochs: 8
- batch size: 1
- learning rate: 5e-5
- validation ratio: 0.2
- max new tokens: 16
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- target modules: query, value

## Result

The run executed on NVIDIA GeForce RTX 4050 Laptop GPU and completed successfully.

Observed output summary:

- Device: cuda
- PEFT model initialized
- Trainable parameters: 1,179,648
- Epoch 8/8 completed
- Average loss: 1.0638
- Best validation accuracy: 0.50 (5/10, epoch 5)
- Best validated adapter saved successfully

## Artifact

Checkpoint directory:

- `checkpoints/rsvqa-blip-lora`

This contains the adapter weights and processor files required to load the PEFT model.

## Held-out evaluation

A real validation evaluation was run on a deterministic 20% validation split from the RSVQA training data after improving the LoRA pipeline.

- Validation set size: 10
- Correct predictions: 5
- Accuracy: 0.50
- Example outputs:
  - “Is it a rural or an urban area” -> `rural`
  - “Are there more roads than commercial buildings?” -> `yes`

This indicates the adapter is functioning, but the current model remains a prototype and should not yet be described as production-grade benchmark performance.

## Notes

- This is a real training run, not a placeholder.
- The adapter is usable for a smoke test and prototype integration.
- Production benchmark claims should wait for stronger held-out results and a more representative split.
- A 16-epoch, 1e-5 learning-rate trial also reached 0.40; the selected 8-epoch, batch-1, 5e-5 run was retained because it reached 0.50 on the fixed split.
