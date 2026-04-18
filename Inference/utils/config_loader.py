"""
YAML configuration loader with path validation and friendly errors.
"""

import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and validate config.yaml.

    Args:
        config_path: Explicit path to config file.
                     Defaults to Inference/config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        SystemExit: On missing file or invalid structure (prints friendly message).
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        _fail(f"Configuration file not found: {path.resolve()}\n"
              "Please ensure config.yaml exists in the Inference/ directory.")

    with open(path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            _fail(f"Failed to parse config.yaml:\n{e}")

    if cfg is None:
        _fail("config.yaml is empty.")

    _apply_defaults(cfg)
    _validate(cfg, path.parent)
    return cfg


def resolve_model_path(cfg: dict, model_key: str) -> Path:
    """Resolve a model path relative to the Inference/ directory."""
    base = Path(__file__).parent.parent
    raw = cfg["models"][model_key]["path"]
    resolved = (base / raw).resolve()
    if not resolved.exists():
        _fail(f"Model file not found: {resolved}\n"
              f"Expected path from config key 'models.{model_key}.path': {raw}")
    return resolved


def resolve_sahi_model_path(cfg: dict, model_key: str) -> Path | None:
    """Resolve SAHI model path with fallback order.

    Priority:
      1) sahi.model_path (global override)
      2) models.<key>.sahi_path
      3) models.<key>.path with .pt sibling fallback when possible

    Returns:
        Resolved model path or None if no valid SAHI path is found.
    """
    base = Path(__file__).parent.parent

    global_override = cfg.get("sahi", {}).get("model_path")
    if global_override:
        resolved = (base / global_override).resolve()
        if resolved.exists():
            return resolved
        return None

    model_cfg = cfg.get("models", {}).get(model_key, {})
    model_override = model_cfg.get("sahi_path")
    if model_override:
        resolved = (base / model_override).resolve()
        if resolved.exists():
            return resolved

    standard_raw = model_cfg.get("path")
    if not standard_raw:
        return None

    standard_path = (base / standard_raw).resolve()
    if standard_path.suffix.lower() == ".onnx":
        pt_candidate = standard_path.with_suffix(".pt")
        if pt_candidate.exists():
            return pt_candidate

    if standard_path.exists():
        return standard_path

    return None


def _apply_defaults(cfg: dict) -> None:
    """Populate optional sections with safe runtime defaults."""
    defaults = cfg.setdefault("defaults", {})
    defaults.setdefault("confidence_threshold", 0.25)
    defaults.setdefault("iou_threshold", 0.45)
    defaults.setdefault("input_size", 640)
    defaults.setdefault("camera_index", 0)
    defaults.setdefault("camera_scan_max", 5)

    inference = cfg.setdefault("inference", {})
    inference.setdefault("mode", "standard")
    inference.setdefault("enable_sahi_for_live_sources", False)
    inference.setdefault("sahi_live_every_n_frames", 5)
    inference.setdefault("fallback_to_standard_on_error", True)

    sahi = cfg.setdefault("sahi", {})
    sahi.setdefault("enabled", True)
    sahi.setdefault("model_path", None)
    sahi.setdefault("slice_height", 640)
    sahi.setdefault("slice_width", 640)
    sahi.setdefault("overlap_height_ratio", 0.2)
    sahi.setdefault("overlap_width_ratio", 0.2)
    sahi.setdefault("postprocess_type", "NMM")
    sahi.setdefault("postprocess_match_metric", "IOS")
    sahi.setdefault("postprocess_match_threshold", 0.5)


def _validate(cfg: dict, base: Path) -> None:
    """Run structural validation on config."""
    if "models" not in cfg or not isinstance(cfg["models"], dict):
        _fail("config.yaml must contain a 'models' section.")
    if "class_names" not in cfg or not isinstance(cfg["class_names"], list):
        _fail("config.yaml must contain a 'class_names' list.")
    if len(cfg["class_names"]) == 0:
        _fail("config.yaml 'class_names' list must not be empty.")
    if "defaults" not in cfg:
        _fail("config.yaml must contain a 'defaults' section.")

    mode = str(cfg.get("inference", {}).get("mode", "standard")).lower()
    if mode not in {"standard", "sahi"}:
        _fail("inference.mode must be either 'standard' or 'sahi'.")

    sahi = cfg.get("sahi", {})
    for key in ("slice_height", "slice_width"):
        try:
            value = int(sahi.get(key, 640))
        except (TypeError, ValueError):
            _fail(f"sahi.{key} must be an integer.")
        if value <= 0:
            _fail(f"sahi.{key} must be > 0.")

    for key in ("overlap_height_ratio", "overlap_width_ratio"):
        try:
            value = float(sahi.get(key, 0.2))
        except (TypeError, ValueError):
            _fail(f"sahi.{key} must be a float in [0, 1).")
        if value < 0.0 or value >= 1.0:
            _fail(f"sahi.{key} must be in [0, 1).")

    try:
        n_frames = int(cfg.get("inference", {}).get("sahi_live_every_n_frames", 5))
    except (TypeError, ValueError):
        _fail("inference.sahi_live_every_n_frames must be an integer >= 1.")
    if n_frames < 1:
        _fail("inference.sahi_live_every_n_frames must be >= 1.")


def _fail(message: str) -> None:
    """Print a user-friendly error and exit."""
    print(f"\n✗ Configuration Error\n{'─' * 40}\n{message}\n", file=sys.stderr)
    raise SystemExit(1)
