from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = LOGS_DIR / "train_runs"
EXPERIMENT_RUNS_DIR = LOGS_DIR / "experiments"

for directory in [MODELS_DIR, LOGS_DIR, EXPERIMENTS_DIR, RUNS_DIR, EXPERIMENT_RUNS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


DEFAULT_DATA_YAML = DATASET_DIR / "data.yaml"
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"

# Training defaults tuned as a practical baseline for small-object SAR experiments.
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 960
DEFAULT_PATIENCE = 30
DEFAULT_WORKERS = 2
DEFAULT_DEVICE = "auto"

# Evaluation defaults
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
DEFAULT_MAX_DET = 300

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def timestamp_id() -> str:
    """UTC timestamp used for reproducible run names and report folders."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve paths consistently from project root unless already absolute."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    base = Path(base_dir).expanduser().resolve() if base_dir else PROJECT_ROOT
    return (base / path).resolve()


def resolve_device(preferred: str = "auto") -> str:
    """Resolve runtime device with fallback order: CUDA -> MPS -> CPU."""
    requested = (preferred or "auto").strip().lower()
    if requested != "auto":
        return requested

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"