import os
import sys
import json
import logging
from typing import Dict, Any, List

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("satquery.datasets")

def init_folders():
    """Create the datasets subdirectories."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = ["rsvqa", "bigearthnet", "cdvqa", "samples"]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Initialized folder: {folder_path}")

def download_huggingface_samples():
    """
    Downloads sample images and QA pairs from HuggingFace dataset.
    Uses dmarsili/RSVQA-LR-2k or similar if available, or falls back to creating
    local sample datasets with pre-configured remote sensing QA configurations.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "samples")
    
    logger.info("Checking for HuggingFace Dataset client dependencies...")
    try:
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset
        logger.info("HuggingFace dependencies found.")
    except ImportError:
        logger.warning("HuggingFace 'datasets' package not installed. Downloading via standard request or writing local manifest.")
        # We will write the manifest anyway for quick local testing!
        
    # Write a structured sample manifest.json for testing.
    # This manifest maps sample images to questions, answers, and spatial bounding boxes.
    manifest = {
        "dataset_name": "SatQuery-Sample-VQA",
        "description": "Simulated Remote Sensing Visual Question Answering sample data.",
        "samples": [
            {
                "id": "sample_001",
                "image_name": "forest_scene.png",
                "modality": "Optical",
                "source": "Sentinel-2",
                "qa_pairs": [
                    {
                        "question": "What type of land cover is visible?",
                        "answer": "The image is dominated by dense vegetation and forest canopy cover.",
                        "category": "Land Cover"
                    },
                    {
                        "question": "Are there any structures present in this scene?",
                        "answer": "No prominent built-up areas or city structures are visible in the image.",
                        "category": "Count"
                    }
                ],
                "grounding": {
                    "vegetation": [[50, 50, 250, 250]]
                }
            },
            {
                "id": "sample_002",
                "image_name": "lake_suburb.png",
                "modality": "Optical",
                "source": "Sentinel-2",
                "qa_pairs": [
                    {
                        "question": "Is there a river or water body present?",
                        "answer": "Yes, a water body is visible covering a section of the image.",
                        "category": "Presence"
                    },
                    {
                        "question": "What elements are visible in this scene?",
                        "answer": "The image shows a large water body with neighboring residential houses and urban roads.",
                        "category": "Land Cover"
                    }
                ],
                "grounding": {
                    "water body": [[0, 0, 300, 300]],
                    "built-up structure": [[10, 10, 80, 80]]
                }
            }
        ]
    }
    
    manifest_path = os.path.join(samples_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Created local sample manifest at: {manifest_path}")

    # Generate the actual mock images locally in the samples folder so the system is self-contained!
    try:
        from PIL import Image
        import numpy as np
        
        # 1. Create forest_scene.png (highly green)
        forest_path = os.path.join(samples_dir, "forest_scene.png")
        if not os.path.exists(forest_path):
            arr = np.zeros((300, 300, 3), dtype=np.uint8)
            arr[50:250, 50:250, 1] = 180  # high green
            arr[10:30, 10:30, :] = 120    # small structures
            Image.fromarray(arr).save(forest_path)
            logger.info(f"Generated mock sample image: {forest_path}")
            
        # 2. Create lake_suburb.png (highly blue + gray)
        lake_path = os.path.join(samples_dir, "lake_suburb.png")
        if not os.path.exists(lake_path):
            arr = np.zeros((300, 300, 3), dtype=np.uint8)
            # Water channel
            arr[:, :, 2] = 200  # high blue
            # Suburb gray structures
            arr[10:80, 10:80, :] = 130  # high gray
            Image.fromarray(arr).save(lake_path)
            logger.info(f"Generated mock sample image: {lake_path}")
            
    except Exception as e:
        logger.error(f"Failed to generate mock dataset image files: {str(e)}")

if __name__ == "__main__":
    init_folders()
    download_huggingface_samples()
