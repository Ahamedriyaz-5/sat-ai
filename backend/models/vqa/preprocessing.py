"""Input preparation for the RGB-only RSVQA BLIP LoRA adapter."""

from pathlib import Path

from PIL import Image


def load_rgb_image(image_path: str) -> Image.Image:
    """Load an RGB visualization compatible with the model's training inputs.

    The adapter was trained with ``PIL.Image.open(...).convert("RGB")`` on
    RSVQA imagery. It consequently accepts PNG/JPEG and RGB/RGBA TIFF images,
    but not raw multispectral or SAR tensors. The saved BLIP processor performs
    its own resize and normalization after this function returns.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as source:
            bands = len(source.getbands())
            if bands > 4:
                raise ValueError(
                    "The fine-tuned RSVQA BLIP adapter expects an RGB image, not a raw multispectral raster. "
                    "Provide an RGB visualization or route multispectral classification to /land-cover."
                )
            return source.convert("RGB").copy()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to prepare an RGB image for VQA: {exc}") from exc
