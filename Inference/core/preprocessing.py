"""
Image preprocessing pipeline for YOLO ONNX inference.

Handles:
- Letterbox resizing (preserves aspect ratio with padding)
- BGR → RGB color conversion
- [0, 255] → [0.0, 1.0] normalization
- HWC → NCHW transposition
- Batch dimension expansion

Returns scale/padding metadata so postprocessing can map
bounding boxes back to original image coordinates.
"""

import cv2
import numpy as np


class PreprocessResult:
    """Holds the preprocessed tensor and inverse-transform metadata."""
    __slots__ = ("tensor", "scale", "pad_x", "pad_y", "orig_shape")

    def __init__(self, tensor: np.ndarray, scale: float,
                 pad_x: int, pad_y: int, orig_shape: tuple[int, int]):
        self.tensor = tensor          # (1, 3, H, W) float32
        self.scale = scale            # Resize scale factor
        self.pad_x = pad_x            # Left padding in pixels
        self.pad_y = pad_y            # Top padding in pixels
        self.orig_shape = orig_shape  # (orig_h, orig_w)


def preprocess(frame: np.ndarray, input_size: int = 640) -> PreprocessResult:
    """Preprocess a BGR frame for YOLO inference.

    Args:
        frame:      Raw BGR image from OpenCV (H, W, 3).
        input_size: Target square size the model expects (default 640).

    Returns:
        PreprocessResult with the tensor and coordinate mapping info.
    """
    orig_h, orig_w = frame.shape[:2]

    # ── Letterbox resize ───────────────────────────────────────────────
    scale = min(input_size / orig_h, input_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # ── Color + normalize + transpose ──────────────────────────────────
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))  # HWC → CHW
    batched = np.expand_dims(transposed, axis=0)       # Add batch dim

    return PreprocessResult(
        tensor=np.ascontiguousarray(batched),
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_shape=(orig_h, orig_w),
    )
