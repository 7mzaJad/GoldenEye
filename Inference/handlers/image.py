"""
Image upload and single-image prediction handler.

Loads an image file, runs it through the inference pipeline,
and returns annotated results.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from core.engine import InferenceEngine
from utils.drawing import Detection, draw_detections

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class ImageHandler:
    """Handles loading and running inference on uploaded images."""

    def __init__(self, engine: InferenceEngine):
        self._engine = engine

    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        inference_mode: str = "standard",
    ) -> tuple[np.ndarray, np.ndarray, list[Detection], float, dict]:
        """Load an image and run detection.

        Args:
            image_path:      Path to the image file.
            conf_threshold:  Confidence threshold.
            iou_threshold:   IoU threshold for NMS.

        Returns:
            Tuple of (original_frame, annotated_frame, detections, inference_ms, meta).

        Raises:
            ValueError: If file type is unsupported or file can't be read.
        """
        path = Path(image_path)

        # Validate extension
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}\n"
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Read image
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(
                f"Failed to read image: {path.name}\n"
                "The file may be corrupted or not a valid image."
            )

        # Run inference
        detections, inference_ms, meta = self._engine.predict(
            frame,
            conf_threshold,
            iou_threshold,
            mode=inference_mode,
            source_type="image",
            return_meta=True,
        )

        # Draw results on a copy
        annotated = frame.copy()
        draw_detections(annotated, detections)

        logger.info(
            "Image prediction: file=%s requested_mode=%s actual_mode=%s "
            "latency_ms=%.2f detections=%d resolution=%sx%s conf=%.2f",
            path.name,
            meta.get("requested_mode"),
            meta.get("actual_mode"),
            inference_ms,
            len(detections),
            frame.shape[1],
            frame.shape[0],
            conf_threshold,
        )

        return frame, annotated, detections, inference_ms, meta
