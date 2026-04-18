"""
SAHI inference module.

This module is intentionally separate from the standard ONNX path so that:
- SAHI stays optional and inference-only.
- The existing real-time pipeline remains intact.
- Both paths return the same Detection structure for stable rendering/UI code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from utils.drawing import Detection

logger = logging.getLogger(__name__)


class SahiInferenceEngine:
    """Runs sliced inference with SAHI and converts outputs to Detection."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str],
        settings: dict[str, Any],
    ):
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.settings = settings

        self._detection_model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Initialize SAHI detection model once for reuse across predictions."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"SAHI model file not found: {self.model_path.resolve()}")

        try:
            from sahi import AutoDetectionModel
        except Exception as exc:
            raise RuntimeError(
                "SAHI is not available. Install dependencies from Inference/requirements.txt"
            ) from exc

        model_type_candidates = self._resolve_model_type_candidates(self.model_path)
        confidence_threshold = float(self.settings.get("confidence_threshold", 0.25))

        last_error: Exception | None = None
        for model_type in model_type_candidates:
            try:
                self._detection_model = AutoDetectionModel.from_pretrained(
                    model_type=model_type,
                    model_path=str(self.model_path),
                    confidence_threshold=confidence_threshold,
                    image_size=int(self.settings.get("slice_height", 640)),
                )
                self._is_loaded = True
                logger.info(
                    "SAHI model initialized: type=%s path=%s",
                    model_type,
                    self.model_path.name,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.debug("SAHI model init failed for model_type=%s: %s", model_type, exc)

        raise RuntimeError(
            "Failed to initialize SAHI detection model. "
            f"Tried model types: {model_type_candidates}. Last error: {last_error}"
        )

    def unload(self) -> None:
        self._detection_model = None
        self._is_loaded = False

    def predict(self, frame: np.ndarray, conf_threshold: float) -> list[Detection]:
        """Run sliced inference and map SAHI predictions into unified Detection objects."""
        if not self._is_loaded or self._detection_model is None:
            return []

        try:
            from sahi.predict import get_sliced_prediction
        except Exception as exc:
            raise RuntimeError("Failed to import SAHI sliced prediction API.") from exc

        # SAHI can improve recall for small/far objects by processing overlapping tiles,
        # but this usually adds latency compared with standard full-frame inference.
        result = get_sliced_prediction(
            image=frame,
            detection_model=self._detection_model,
            slice_height=int(self.settings.get("slice_height", 640)),
            slice_width=int(self.settings.get("slice_width", 640)),
            overlap_height_ratio=float(self.settings.get("overlap_height_ratio", 0.2)),
            overlap_width_ratio=float(self.settings.get("overlap_width_ratio", 0.2)),
            postprocess_type=str(self.settings.get("postprocess_type", "NMM")),
            postprocess_match_metric=str(self.settings.get("postprocess_match_metric", "IOS")),
            postprocess_match_threshold=float(
                self.settings.get("postprocess_match_threshold", 0.5)
            ),
            verbose=0,
        )

        detections: list[Detection] = []
        for pred in result.object_prediction_list:
            score = float(pred.score.value)
            if score < conf_threshold:
                continue

            cid = int(pred.category.id)
            class_name = pred.category.name
            if not class_name:
                class_name = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"

            bbox = pred.bbox
            detections.append(
                Detection(
                    bbox=(
                        int(bbox.minx),
                        int(bbox.miny),
                        int(bbox.maxx),
                        int(bbox.maxy),
                    ),
                    confidence=score,
                    class_id=cid,
                    class_name=class_name,
                )
            )

        return detections

    def settings_for_logs(self) -> dict[str, Any]:
        """Return concise SAHI runtime settings for mode-aware logging."""
        return {
            "slice_height": int(self.settings.get("slice_height", 640)),
            "slice_width": int(self.settings.get("slice_width", 640)),
            "overlap_height_ratio": float(self.settings.get("overlap_height_ratio", 0.2)),
            "overlap_width_ratio": float(self.settings.get("overlap_width_ratio", 0.2)),
            "postprocess_type": str(self.settings.get("postprocess_type", "NMM")),
            "postprocess_match_metric": str(self.settings.get("postprocess_match_metric", "IOS")),
            "postprocess_match_threshold": float(
                self.settings.get("postprocess_match_threshold", 0.5)
            ),
        }

    @staticmethod
    def _resolve_model_type_candidates(path: Path) -> list[str]:
        """Choose SAHI model_type candidates based on file suffix.

        We try multiple options to stay robust across SAHI versions and weight formats.
        """
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            return ["yolov8onnx", "ultralytics"]
        if suffix == ".pt":
            return ["ultralytics", "yolov8"]
        return ["ultralytics"]
