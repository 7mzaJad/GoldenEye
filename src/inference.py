"""Inference helpers for Ultralytics YOLO models."""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from config import (
    BEST_MODEL_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    PREDICTIONS_DIR,
)
from preprocessing import list_image_files


class YOLOInference:
    """Convenience wrapper for single-image and batch YOLO predictions."""

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        selected = model_path or BEST_MODEL_PATH
        selected_path = Path(selected).expanduser()
        if selected_path.exists():
            self.model = YOLO(str(selected_path.resolve()))
        else:
            # Allow official model IDs such as "yolov8n.pt".
            self.model = YOLO(str(selected))

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @staticmethod
    def _summarize_result(result: Any) -> dict[str, Any]:
        detections = int(len(result.boxes)) if result.boxes is not None else 0
        top_confidence = 0.0
        if detections > 0 and getattr(result.boxes, "conf", None) is not None:
            top_confidence = float(result.boxes.conf.max().item())

        return {
            "image_path": str(result.path),
            "detections": detections,
            "top_confidence": top_confidence,
        }

    @staticmethod
    def _to_list(tensor_like: Any) -> list[Any]:
        if tensor_like is None:
            return []

        value = tensor_like
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def parse_result_detailed(self, result: Any) -> dict[str, Any]:
        """Return normalized detection details for API responses."""
        names_lookup = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)

        xyxy_values = self._to_list(getattr(boxes, "xyxy", None))
        conf_values = self._to_list(getattr(boxes, "conf", None))
        cls_values = self._to_list(getattr(boxes, "cls", None))

        detections: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}

        max_len = min(len(xyxy_values), len(conf_values), len(cls_values))
        for index in range(max_len):
            raw_class_id = int(cls_values[index])
            class_name = str(names_lookup.get(raw_class_id, raw_class_id))
            confidence = float(conf_values[index])
            x1, y1, x2, y2 = [float(coord) for coord in xyxy_values[index]]

            detections.append(
                {
                    "class_id": raw_class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        top_confidence = max((item["confidence"] for item in detections), default=0.0)
        speed = getattr(result, "speed", {}) or {}

        return {
            "image_path": str(getattr(result, "path", "")),
            "detections": len(detections),
            "top_confidence": top_confidence,
            "latency_ms": float(speed.get("inference", 0.0)),
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
            "class_counts": class_counts,
            "items": detections,
        }

    def predict_image(
        self,
        img_path: str,
        save: bool = False,
        conf: float | None = None,
        iou: float | None = None,
    ) -> dict[str, Any]:
        image_path = Path(img_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=conf if conf is not None else self.conf_threshold,
            iou=iou if iou is not None else self.iou_threshold,
            save=save,
            project=str(PREDICTIONS_DIR),
            name="predict-single",
            exist_ok=True,
            verbose=False,
        )
        return self._summarize_result(results[0])

    def predict_batch(
        self,
        input_dir: str,
        save: bool = True,
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[dict[str, Any]]:
        image_files = list_image_files(input_dir, recursive=False)
        if not image_files:
            raise ValueError(f"No supported image files found in: {input_dir}")

        results = self.model.predict(
            source=[str(path) for path in image_files],
            conf=conf if conf is not None else self.conf_threshold,
            iou=iou if iou is not None else self.iou_threshold,
            save=save,
            project=str(PREDICTIONS_DIR),
            name="predict-batch",
            exist_ok=True,
            verbose=False,
        )
        return [self._summarize_result(result) for result in results]

    def predict_array(
        self,
        image_bgr: Any,
        conf: float | None = None,
        iou: float | None = None,
    ) -> dict[str, Any]:
        """Run inference on a BGR numpy array and return detailed metrics."""
        results = self.model.predict(
            source=image_bgr,
            conf=conf if conf is not None else self.conf_threshold,
            iou=iou if iou is not None else self.iou_threshold,
            save=False,
            verbose=False,
        )
        return self.parse_result_detailed(results[0])

    def predict_array_with_preview(
        self,
        image_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Run inference on a BGR image and return metrics plus annotated frame."""
        results = self.model.predict(
            source=image_bgr,
            conf=conf if conf is not None else self.conf_threshold,
            iou=iou if iou is not None else self.iou_threshold,
            save=False,
            verbose=False,
        )
        result = results[0]
        parsed = self.parse_result_detailed(result)
        annotated = result.plot()
        return parsed, annotated
