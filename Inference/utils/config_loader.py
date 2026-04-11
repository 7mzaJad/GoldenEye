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


def _fail(message: str) -> None:
    """Print a user-friendly error and exit."""
    print(f"\n✗ Configuration Error\n{'─' * 40}\n{message}\n", file=sys.stderr)
    raise SystemExit(1)
