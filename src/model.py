"""Model factory helpers for Ultralytics YOLO."""

from ultralytics import YOLO

from config import MODEL_TYPE, MODELS_DIR


def build_model(model_type: str = MODEL_TYPE, weights: str | None = None) -> YOLO:
    """Create a YOLO model from local checkpoints or official pretrained weights."""
    if weights:
        return YOLO(weights)

    normalized = model_type.lower().strip()
    if normalized not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported YOLOv8 model type: {model_type}")

    local_checkpoint = MODELS_DIR / f"Yolov8{normalized}" / "best.pt"
    if local_checkpoint.exists():
        return YOLO(str(local_checkpoint))

    return YOLO(f"yolov8{normalized}.pt")
