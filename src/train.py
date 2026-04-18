from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_PATIENCE,
    DEFAULT_WORKERS,
    RUNS_DIR,
    resolve_device,
    resolve_path,
    timestamp_id,
)
from model import resolve_model_reference

logger = logging.getLogger(__name__)


def _read_last_epoch_metrics(save_dir: Path) -> dict[str, float]:
    """Read final row from Ultralytics results.csv for training summary metrics."""
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    last_row = rows[-1]
    metrics: dict[str, float] = {}
    for key, value in last_row.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def train_yolo(
    data_yaml: str,
    model_weights: str = DEFAULT_MODEL_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    device: str = DEFAULT_DEVICE,
    run_name: str | None = None,
    project_dir: str | Path = RUNS_DIR,
    patience: int = DEFAULT_PATIENCE,
    workers: int = DEFAULT_WORKERS,
    exist_ok: bool = False,
    extra_train_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train YOLO model and return structured metadata for experiment logging."""
    data_yaml_path = resolve_path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    run_id = run_name or f"train_{timestamp_id()}"
    project = resolve_path(project_dir)
    project.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    model_reference = resolve_model_reference(str(model_weights))

    logger.info(
        "Training run started: run=%s model=%s data=%s device=%s",
        run_id,
        model_reference,
        data_yaml_path,
        resolved_device,
    )

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml_path),
        "epochs": int(epochs),
        "imgsz": int(img_size),
        "batch": int(batch_size),
        "device": resolved_device,
        "project": str(project),
        "name": run_id,
        "patience": int(patience),
        "workers": int(workers),
        "exist_ok": bool(exist_ok),
    }
    if extra_train_args:
        train_kwargs.update(extra_train_args)

    model = YOLO(model_reference)

    start_time = time.perf_counter()
    train_result = model.train(**train_kwargs)
    train_seconds = time.perf_counter() - start_time

    save_dir = Path(train_result.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    train_metrics = _read_last_epoch_metrics(save_dir)

    output = {
        "run_name": run_id,
        "model_reference": model_reference,
        "data_yaml": str(data_yaml_path),
        "device": resolved_device,
        "save_dir": str(save_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "last_weights": str(last_weights) if last_weights.exists() else None,
        "train_seconds": round(train_seconds, 4),
        "train_metrics": train_metrics,
    }

    logger.info(
        "Training run finished: run=%s best_weights=%s train_seconds=%.2f",
        run_id,
        output["best_weights"],
        train_seconds,
    )
    return output
