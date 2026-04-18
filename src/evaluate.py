"""Evaluation helpers for Ultralytics YOLO models."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import BEST_MODEL_PATH, DATA_YAML, DEVICE, IMG_SIZE


def evaluate_yolo(
    model_path: str = BEST_MODEL_PATH,
    data_yaml: str = DATA_YAML,
    split: str = "val",
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
) -> dict[str, Any]:
    """Run YOLO validation and return a concise metric dictionary."""
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    dataset_file = Path(data_yaml).expanduser().resolve()
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_file}")

    model = YOLO(str(model_file))
    metrics = model.val(
        data=str(dataset_file),
        split=split,
        imgsz=img_size,
        device=device,
        verbose=False,
    )

    box_metrics = getattr(metrics, "box", None)
    return {
        "map50": float(getattr(box_metrics, "map50", 0.0)),
        "map50_95": float(getattr(box_metrics, "map", 0.0)),
        "precision": float(getattr(box_metrics, "mp", 0.0)),
        "recall": float(getattr(box_metrics, "mr", 0.0)),
    }


if __name__ == "__main__":
    print(evaluate_yolo())
