"""Shared configuration for the root YOLO training and inference pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTIONS_DIR = LOGS_DIR / "predictions"

for directory in (MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR):
    directory.mkdir(exist_ok=True, parents=True)


MODEL_TYPE = "n"
MODEL_NAME = f"yolov8{MODEL_TYPE}"

LOCAL_MODEL_DIR = MODELS_DIR / "Yolov8n"
LOCAL_BEST_MODEL_PATH = LOCAL_MODEL_DIR / "best.pt"

# Prefer local checkpoint if available; otherwise use an official pretrained ID.
MODEL_WEIGHTS = str(
    LOCAL_BEST_MODEL_PATH if LOCAL_BEST_MODEL_PATH.exists() else f"{MODEL_NAME}.pt"
)
BEST_MODEL_PATH = str(LOCAL_BEST_MODEL_PATH)

EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "cpu"

DATA_YAML = str(DATASET_DIR / "data.yaml")

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
