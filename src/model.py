from __future__ import annotations

from pathlib import Path

from config import MODELS_DIR, resolve_path


YOLO_MODEL_REGISTRY = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
}


def list_supported_models() -> list[str]:
    """Return model aliases available for experiment configs and CLI usage."""
    return sorted(YOLO_MODEL_REGISTRY.keys())


def resolve_model_reference(model_name_or_path: str) -> str:
    """Resolve model input to either explicit local path or YOLO model alias weight.

    Behavior:
    - If given path exists locally, return absolute path.
    - If alias exists in registry, return known base weight name.
    - Otherwise return original input so Ultralytics can resolve it.
    """
    maybe_path = Path(model_name_or_path).expanduser()
    if maybe_path.exists():
        return str(resolve_path(maybe_path))

    alias = model_name_or_path.strip().lower()
    if alias in YOLO_MODEL_REGISTRY:
        return YOLO_MODEL_REGISTRY[alias]

    # If user passes direct .pt name or hub reference, let Ultralytics handle it.
    return model_name_or_path
