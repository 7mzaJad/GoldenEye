import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


MODEL_NAME = "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
MODEL_WEIGHTS = str(MODELS_DIR / "yolov8n.pt")  # Pretrained weights
BEST_MODEL_PATH = str(MODELS_DIR / "best.pt")  # Best model path
# LAST_MODEL_PATH = str(MODELS_DIR / "last.pt")  # Last model checkpoint

# Training Parameters
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640  # YOLO standard image size
DEVICE = "cpu"  # Use 'cpu' for macOS without GPU, or 'mps' for Apple Silicon
# WORKERS = 0  # Number of dataloader workers
# PATIENCE = 20  # Early stopping patience
# MOMENTUM = 0.937
# WEIGHT_DECAY = 0.0005
# LR0 = 0.01  # Initial learning rate
# LRF = 0.01  # Final learning rate

# Data Configuration
DATA_YAML = str(DATASET_DIR / "data.yaml")
DATA_CONFIG = {
    "path": str(DATASET_DIR),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,  # Number of classes
    "names": ["human"]  # Class names
}
'''
# Augmentation Configuration
AUGMENTATION_CONFIG = {
    "hsv_h": 0.015,  # HSV-Hue augmentation (fraction)
    "hsv_s": 0.7,    # HSV-Saturation augmentation (fraction)
    "hsv_v": 0.4,    # HSV-Value augmentation (fraction)
    "degrees": 10,   # Rotation degrees
    "translate": 0.1,  # Image translation (fraction)
    "scale": 0.5,    # Image scale (fraction)
    "flipud": 0.0,   # Flip upside-down probability
    "fliplr": 0.5,   # Flip left-right probability
    "mosaic": 1.0,   # Mosaic augmentation probability
    "perspective": 0.0  # Perspective augmentation
}

# Validation Configuration
VAL_BATCH_SIZE = 32
CONF_THRESHOLD = 0.25  # Confidence threshold for predictions
IOU_THRESHOLD = 0.45   # IoU threshold for NMS

# Logging Configuration
VERBOSE = True
SAVE_PERIOD = 10  # Save checkpoint every N epochs
PLOT = True  # Plot results
KEEP_BEST = True  # Keep only best model

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

PREDICT_CONF_THRESHOLD = 0.25
PREDICT_IOU_THRESHOLD = 0.45
SAVE_PREDICTIONS = True  # Save prediction results
SAVE_CROPS = False  # Save cropped predictions
LINE_WIDTH = 2  # Bounding box line width
FONT_SIZE = 0.6  # Label font size
'''