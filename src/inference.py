from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import (
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_MODEL_WEIGHTS,
    LOGS_DIR,
    resolve_device,
    resolve_path,
)


class YOLOInference:
    """Simple inference wrapper used by scripts and quick smoke tests."""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_MODEL_WEIGHTS,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = DEFAULT_DEVICE,
    ):
        weights = str(weights_path)
        if Path(weights_path).expanduser().exists():
            weights = str(resolve_path(weights_path))

        self.model = YOLO(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        self.device = resolve_device(device)

    def predict_image(
        self,
        image_path: str | Path,
        save: bool = False,
        project_dir: str | Path = LOGS_DIR / "predictions",
        run_name: str = "predict",
        extra_predict_args: dict[str, Any] | None = None,
    ):
        """Run prediction on a single image path and return raw Ultralytics results."""
        image = resolve_path(image_path)
        kwargs: dict[str, Any] = {
            "source": str(image),
            "imgsz": self.img_size,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "save": save,
            "project": str(resolve_path(project_dir)),
            "name": run_name,
            "max_det": DEFAULT_MAX_DET,
            "verbose": False,
        }
        if extra_predict_args:
            kwargs.update(extra_predict_args)
        return self.model.predict(**kwargs)

    def predict_batch(
        self,
        source_dir: str | Path,
        save: bool = False,
        project_dir: str | Path = LOGS_DIR / "predictions",
        run_name: str = "predict_batch",
        extra_predict_args: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run prediction on a folder and return compact per-image summaries."""
        source = resolve_path(source_dir)
        kwargs: dict[str, Any] = {
            "source": str(source),
            "imgsz": self.img_size,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "save": save,
            "project": str(resolve_path(project_dir)),
            "name": run_name,
            "max_det": DEFAULT_MAX_DET,
            "verbose": False,
        }
        if extra_predict_args:
            kwargs.update(extra_predict_args)

        raw_results = self.model.predict(**kwargs)
        summaries: list[dict[str, Any]] = []
        for result in raw_results:
            boxes = result.boxes
            summaries.append(
                {
                    "image": str(result.path),
                    "num_detections": int(len(boxes)) if boxes is not None else 0,
                }
            )
        return summaries
