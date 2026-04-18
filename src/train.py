"""Training entrypoints for Ultralytics YOLO."""

from pathlib import Path
import shutil

from ultralytics import YOLO

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATA_YAML,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    LOGS_DIR,
    MODEL_TYPE,
    MODEL_WEIGHTS,
    MODELS_DIR,
)


def _select_training_weights(model_type: str) -> str:
    normalized = model_type.lower().strip()
    if normalized not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported YOLOv8 model type: {model_type}")

    local_candidate = MODELS_DIR / f"Yolov8{normalized}" / "best.pt"
    if local_candidate.exists():
        return str(local_candidate)

    configured = Path(MODEL_WEIGHTS).expanduser()
    if configured.exists():
        return str(configured)

    return f"yolov8{normalized}.pt"


def _find_best_checkpoint(model: YOLO) -> Path | None:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return None

    best = Path(str(getattr(trainer, "best", "")))
    if best.exists():
        return best

    save_dir = Path(str(getattr(trainer, "save_dir", "")))
    fallback = save_dir / "weights" / "best.pt"
    if fallback.exists():
        return fallback

    return None


def train_yolo(
    data_yaml: str = DATA_YAML,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    model_type: str = MODEL_TYPE,
) -> YOLO:
    """Train a YOLO model and return a model loaded from the best checkpoint."""
    data_path = Path(data_yaml).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    weights = _select_training_weights(model_type)
    model = YOLO(weights)

    run_name = f"train_yolov8{model_type.lower().strip()}"
    model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(LOGS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    best_checkpoint = _find_best_checkpoint(model)
    if best_checkpoint is None:
        return model

    canonical_best = Path(BEST_MODEL_PATH).expanduser()
    canonical_best.parent.mkdir(parents=True, exist_ok=True)
    if canonical_best.resolve() != best_checkpoint.resolve():
        shutil.copy2(best_checkpoint, canonical_best)

    return YOLO(str(canonical_best))
