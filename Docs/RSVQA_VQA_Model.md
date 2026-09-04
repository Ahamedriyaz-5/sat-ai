# RSVQA BLIP LoRA VQA model

## Verified checkpoint

SatQuery's VQA specialist uses the locally stored adapter at
`checkpoints/rsvqa-blip-lora` with the base model `Salesforce/blip-vqa-base`.
It is a LoRA adapter (rank 8, alpha 16, dropout 0.05) applied to BLIP's
query/value modules. The selected adapter was trained for eight epochs on the
local RSVQA subset and recorded `5/10` exact-match validation accuracy on its
fixed split. This is a prototype smoke-test result, not a production benchmark.

The adapter is a real remote-sensing VQA adaptation: the training records pair
satellite RGB imagery with RSVQA-style questions and answers. The alternate
`rsvqa-blip-lora-cpu80-1e` checkpoint remains an experiment artifact and is not
the configured production adapter.

## Input and preprocessing contract

Training used `PIL.Image.open(...).convert("RGB")`, followed by the saved BLIP
processor. The processor converts RGB, resizes to 384×384, rescales, and
normalizes. Therefore the model accepts PNG/JPEG and RGB/RGBA TIFF
visualizations. It is not a raw multispectral or SAR VQA model; do not pass
10/12/14-band rasters to this adapter. Use the separate BigEarthNet
`/api/v1/land-cover` specialist for compatible multi-band land-cover inference.

## Runtime configuration

- `VQA_USE_FALLBACK=false` enables BLIP + the LoRA adapter.
- `VQA_USE_FALLBACK=true` selects the deterministic spectral fallback.
- `VQA_LOCAL_FILES_ONLY=true` (default) loads the cached base model without a
  network request. Set it to `false` only when a Hugging Face download is
  intended.
- `VQA_ADAPTER_PATH` selects the adapter directory.

The VQA response and agent execution trace expose `inference_mode` (`model` or
`fallback`) and `fallback_active`, so a fallback answer cannot be presented as
model-backed inference.

## Reproducible smoke test

After the base BLIP model is cached, run:

```bash
.venv/bin/python tests/verify_model_vqa.py
```

The test checks both `/api/v1/vqa` and automatic VQA selection through
`/api/v1/agent`, and fails unless each reports `inference_mode: model`.
