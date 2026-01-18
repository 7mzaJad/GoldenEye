"""
GoldenEye - YOLO Object Detection Pipeline
Main Entry Point

This script provides a unified interface for training and inference
with YOLO models for human detection in SAR (Search and Rescue) tasks.
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE, DATA_YAML,
    LOGS_DIR, MODELS_DIR
)
from train import train_yolo
from inference import YOLOInference
from data_loader import get_dataset_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the GoldenEye pipeline."""
    
    logger.info("=" * 80)
    logger.info("GoldenEye - YOLO Object Detection for SAR")
    logger.info("=" * 80)
    
    # Show dataset information
    logger.info("\nDataset Information:")
    dataset_info = get_dataset_info()
    if dataset_info['is_valid']:
        logger.info(f"✓ Dataset is valid")
        logger.info(f"  Total samples: {dataset_info['samples']['total']}")
        logger.info(f"  Train: {dataset_info['samples']['train']}")
        logger.info(f"  Valid: {dataset_info['samples']['valid']}")
        logger.info(f"  Test: {dataset_info['samples']['test']}")
    else:
        logger.error("✗ Dataset structure is invalid. Please check your dataset.")
        return
    
    # Train YOLO model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING YOLOv8 MODEL")
    logger.info("=" * 80)
    
    try:
        model = train_yolo(
            data_yaml=DATA_YAML,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=DEVICE,
            model_type="n"
        )
        logger.info("✓ Training completed successfully!")
    except Exception as e:
        logger.error(f"✗ Training failed: {str(e)}")
        return
    
    # Run inference on test set
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING INFERENCE ON TEST SET")
    logger.info("=" * 80)
    
    try:
        test_dir = Path(__file__).parent / "Dataset" / "test" / "images"
        if test_dir.exists():
            inference = YOLOInference()
            results = inference.predict_batch(str(test_dir), save=True)
            logger.info(f"✓ Inference completed on {len(results)} test images")
        else:
            logger.warning(f"Test directory not found: {test_dir}")
    except Exception as e:
        logger.error(f"✗ Inference failed: {str(e)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {LOGS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
