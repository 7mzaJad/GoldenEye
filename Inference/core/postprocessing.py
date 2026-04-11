"""
Postprocessing for YOLO ONNX model outputs.

Supports two output formats:
  • YOLOv8 standard:  [1, 5, 8400]  → requires NMS
  • YOLOv26 NMS-free: [1, 300, 6]   → already filtered

All bounding box coordinates are mapped back to the original image frame
using the PreprocessResult metadata.
"""

import cv2
import numpy as np

from core.preprocessing import PreprocessResult
from utils.drawing import Detection


def postprocess(
    output: np.ndarray,
    preprocess_info: PreprocessResult,
    class_names: list[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_type: str = "yolov8",
) -> list[Detection]:
    """Route to the correct postprocessor based on model type.

    Args:
        output:          Raw model output tensor.
        preprocess_info: Metadata from preprocessing (scale, padding).
        class_names:     List of class name strings.
        conf_threshold:  Minimum confidence to keep a detection.
        iou_threshold:   IoU threshold for NMS (YOLOv8 only).
        model_type:      "yolov8" or "yolov26".

    Returns:
        List of Detection objects in original image coordinates.
    """
    if model_type == "yolov26":
        return _postprocess_yolov26(
            output, preprocess_info, class_names, conf_threshold
        )
    else:
        return _postprocess_yolov8(
            output, preprocess_info, class_names, conf_threshold, iou_threshold
        )


# ── YOLOv8 Standard: [1, num_classes+4, num_anchors] ──────────────────────

def _postprocess_yolov8(
    output: np.ndarray,
    info: PreprocessResult,
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Decode YOLOv8 output [1, 4+nc, 8400] → detections."""

    # output shape: (1, 4+nc, 8400) → transpose to (8400, 4+nc)
    preds = output[0].T  # (8400, 5) for single-class

    num_classes = preds.shape[1] - 4
    boxes_cxcywh = preds[:, :4]            # (cx, cy, w, h)
    class_scores = preds[:, 4:]            # (8400, nc)

    # Best class per anchor
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Confidence filter
    mask = confidences >= conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert center-format to corner-format
    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)

    # Map back to original image coordinates
    boxes_orig = _map_to_original(boxes_xyxy, info)

    # NMS via OpenCV
    boxes_xywh_cv = []
    for b in boxes_xyxy:
        boxes_xywh_cv.append([float(b[0]), float(b[1]),
                              float(b[2] - b[0]), float(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh_cv,
        confidences.tolist(),
        conf_threshold,
        iou_threshold,
    )

    if len(indices) == 0:
        return []

    indices = indices.flatten()
    detections = []
    for i in indices:
        x1, y1, x2, y2 = boxes_orig[i]
        cid = int(class_ids[i])
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        detections.append(Detection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(confidences[i]),
            class_id=cid,
            class_name=name,
        ))

    return detections


# ── YOLOv26 NMS-Free: [1, 300, 6] ────────────────────────────────────────

def _postprocess_yolov26(
    output: np.ndarray,
    info: PreprocessResult,
    class_names: list[str],
    conf_threshold: float,
) -> list[Detection]:
    """Decode YOLOv26 NMS-free output [1, 300, 6] → detections.

    Each row: [x1, y1, x2, y2, confidence, class_id]
    Coordinates are in the input_size (640) space.
    """
    preds = output[0]  # (300, 6)

    confidences = preds[:, 4]
    mask = confidences >= conf_threshold
    preds = preds[mask]

    if len(preds) == 0:
        return []

    boxes_xyxy = preds[:, :4]
    confidences = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # Map to original image coordinates
    boxes_orig = _map_to_original(boxes_xyxy, info)

    detections = []
    for i in range(len(preds)):
        x1, y1, x2, y2 = boxes_orig[i]
        cid = int(class_ids[i])
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        detections.append(Detection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(confidences[i]),
            class_id=cid,
            class_name=name,
        ))

    return detections


# ── Coordinate Utilities ──────────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def _map_to_original(
    boxes: np.ndarray, info: PreprocessResult
) -> np.ndarray:
    """Map boxes from input_size space back to original image space.

    Reverses: pad → resize (letterbox).
    """
    mapped = boxes.copy()

    # Remove padding offset
    mapped[:, 0] -= info.pad_x
    mapped[:, 1] -= info.pad_y
    mapped[:, 2] -= info.pad_x
    mapped[:, 3] -= info.pad_y

    # Reverse resize scaling
    mapped /= info.scale

    # Clip to original image bounds
    orig_h, orig_w = info.orig_shape
    mapped[:, 0] = np.clip(mapped[:, 0], 0, orig_w)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, orig_h)
    mapped[:, 2] = np.clip(mapped[:, 2], 0, orig_w)
    mapped[:, 3] = np.clip(mapped[:, 3], 0, orig_h)

    return mapped
