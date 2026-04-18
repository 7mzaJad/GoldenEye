"""
YAML configuration loader with path validation and friendly errors.
"""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration files are missing or invalid."""


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and validate config.yaml.

    Args:
        config_path: Explicit path to config file.
                     Defaults to Inference/config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigError: On missing file or invalid structure.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        _fail(
            f"Configuration file not found: {path.resolve()}\n"
            "Please ensure config.yaml exists in the Inference/ directory."
        )

    with open(path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            _fail(f"Failed to parse config.yaml:\n{e}")

    _validate(cfg)
    return cfg


def resolve_model_path(cfg: dict[str, Any], model_key: str) -> Path:
    """Resolve a model path relative to the Inference/ directory."""
    base = Path(__file__).parent.parent
    try:
        raw = cfg["models"][model_key]["path"]
    except KeyError:
        _fail(f"Missing configuration key: models.{model_key}.path")

    resolved = (base / raw).resolve()
    if not resolved.exists():
        _fail(
            f"Model file not found: {resolved}\n"
            f"Expected path from config key 'models.{model_key}.path': {raw}"
        )
    return resolved


def _validate(cfg: Any) -> None:
    """Run structural validation on config."""
    if not isinstance(cfg, dict):
        _fail("config.yaml root must be a dictionary of settings.")

    if "models" not in cfg or not isinstance(cfg["models"], dict):
        _fail("config.yaml must contain a 'models' section.")
    if "class_names" not in cfg or not isinstance(cfg["class_names"], list):
        _fail("config.yaml must contain a 'class_names' list.")
    if len(cfg["class_names"]) == 0:
        _fail("config.yaml 'class_names' list must not be empty.")
    if "defaults" not in cfg or not isinstance(cfg["defaults"], dict):
        _fail("config.yaml must contain a 'defaults' section.")


def _fail(message: str) -> None:
    """Raise a configuration exception with a user-friendly message."""
    raise ConfigError(message)
