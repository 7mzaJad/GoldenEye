"""
Bounding box and label rendering utilities.

Draws detection results onto OpenCV frames with:
- Per-class color coding
- Semi-transparent label backgrounds
- Readable font sizing scaled to frame resolution
"""

import cv2
import numpy as np

from utils.constants import (
    DETECTION_COLORS,
    BOX_THICKNESS,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LABEL_PADDING,
    LABEL_BG_ALPHA,
)


class Detection:
    """Single detection result."""
    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(self, bbox: tuple[int, int, int, int], confidence: float,
                 class_id: int, class_name: str):
        self.bbox = bbox              # (x1, y1, x2, y2) in original image coords
        self.confidence = confidence   # 0.0–1.0
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"bbox={self.bbox})")


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    font_scale: float | None = None,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on a frame.

    Args:
        frame:      BGR image (will be modified in-place).
        detections: List of Detection objects.
        font_scale: Override default label font scale.
        thickness:  Override default box thickness.

    Returns:
        The annotated frame (same reference, modified in-place).
    """
    if not detections:
        return frame

    h, w = frame.shape[:2]
    # Scale rendering parameters based on frame resolution
    scale_factor = min(w, h) / 640.0
    fs = font_scale if font_scale else max(LABEL_FONT_SCALE * scale_factor, 0.35)
    tk = thickness if thickness else max(int(BOX_THICKNESS * scale_factor), 1)
    pad = max(int(LABEL_PADDING * scale_factor), 2)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = DETECTION_COLORS[det.class_id % len(DETECTION_COLORS)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, tk)

        # Label text
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, LABEL_THICKNESS
        )

        # Label background position — prefer above the box
        label_y1 = y1 - th - 2 * pad
        label_y2 = y1
        if label_y1 < 0:  # Flip below if box is at top edge
            label_y1 = y2
            label_y2 = y2 + th + 2 * pad

        label_x1 = x1
        label_x2 = x1 + tw + 2 * pad

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.addWeighted(overlay, LABEL_BG_ALPHA, frame, 1 - LABEL_BG_ALPHA, 0, frame)

        # Text (dark on bright, white on dark backgrounds)
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
        text_y = label_y2 - pad if label_y1 < y1 else label_y2 - pad
        cv2.putText(frame, label, (label_x1 + pad, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, LABEL_THICKNESS,
                    cv2.LINE_AA)

    return frame
