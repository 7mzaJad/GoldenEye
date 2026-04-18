from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    resolve_device,
    resolve_path,
)

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_model(
    weights_path: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    extra_eval_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate model on a split and return normalized metrics for comparisons."""
    weights = resolve_path(weights_path)
    data = resolve_path(data_yaml)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    resolved_device = resolve_device(device)

    val_kwargs: dict[str, Any] = {
        "data": str(data),
        "split": split,
        "imgsz": int(img_size),
        "batch": int(batch_size),
        "device": resolved_device,
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "verbose": False,
    }
    if extra_eval_args:
        val_kwargs.update(extra_eval_args)

    model = YOLO(str(weights))
    metrics_obj = model.val(**val_kwargs)
    metrics_dict = dict(getattr(metrics_obj, "results_dict", {}) or {})
    speed_dict = dict(getattr(metrics_obj, "speed", {}) or {})

    # Ultralytics metrics keys are version-dependent, so we try dict first then box attrs.
    box = getattr(metrics_obj, "box", None)
    precision = _safe_float(metrics_dict.get("metrics/precision(B)"))
    recall = _safe_float(metrics_dict.get("metrics/recall(B)"))
    map50 = _safe_float(metrics_dict.get("metrics/mAP50(B)"))
    map50_95 = _safe_float(metrics_dict.get("metrics/mAP50-95(B)"))

    if box is not None:
        precision = precision if precision is not None else _safe_float(getattr(box, "mp", None))
        recall = recall if recall is not None else _safe_float(getattr(box, "mr", None))
        map50 = map50 if map50 is not None else _safe_float(getattr(box, "map50", None))
        map50_95 = map50_95 if map50_95 is not None else _safe_float(getattr(box, "map", None))

    result = {
        "weights_path": str(weights),
        "data_yaml": str(data),
        "split": split,
        "device": resolved_device,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "fitness": _safe_float(getattr(metrics_obj, "fitness", None)),
        "speed_preprocess_ms": _safe_float(speed_dict.get("preprocess")),
        "speed_inference_ms": _safe_float(speed_dict.get("inference")),
        "speed_postprocess_ms": _safe_float(speed_dict.get("postprocess")),
        "raw_metrics": {k: _safe_float(v) for k, v in metrics_dict.items()},
    }

    logger.info(
        "Evaluation complete: split=%s map50=%.4f map50_95=%.4f precision=%.4f recall=%.4f",
        split,
        result["map50"] if result["map50"] is not None else -1.0,
        result["map50_95"] if result["map50_95"] is not None else -1.0,
        result["precision"] if result["precision"] is not None else -1.0,
        result["recall"] if result["recall"] is not None else -1.0,
    )
    return result

