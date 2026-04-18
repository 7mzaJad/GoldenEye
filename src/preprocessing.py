from __future__ import annotations


def small_object_train_overrides(img_size: int = 960) -> dict:
    """Recommended Ultralytics overrides for small/distant SAR targets.

    These are conservative defaults for experimentation, not hard constraints.
    """
    return {
        "imgsz": int(img_size),
        "close_mosaic": 10,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.10,
        "degrees": 5.0,
        "translate": 0.10,
        "scale": 0.50,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    }
