from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config import DEFAULT_DATA_YAML, IMAGE_EXTENSIONS, resolve_path


def load_dataset_config(data_yaml: str | Path = DEFAULT_DATA_YAML) -> dict[str, Any]:
    """Load YOLO dataset YAML and return parsed config dict."""
    yaml_path = resolve_path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset YAML structure in: {yaml_path}")

    return config


def _resolve_split_path(raw_path: str | None, yaml_dir: Path) -> Path | None:
    if not raw_path:
        return None

    split_path = Path(str(raw_path)).expanduser()
    if split_path.is_absolute():
        return split_path
    return (yaml_dir / split_path).resolve()


def _count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def get_dataset_info(data_yaml: str | Path = DEFAULT_DATA_YAML) -> dict[str, Any]:
    """Return dataset validity, split paths, class metadata, and sample counts."""
    yaml_path = resolve_path(data_yaml)
    config = load_dataset_config(yaml_path)

    yaml_dir = yaml_path.parent
    train_path = _resolve_split_path(config.get("train"), yaml_dir)
    val_path = _resolve_split_path(config.get("val"), yaml_dir)
    test_path = _resolve_split_path(config.get("test"), yaml_dir)

    missing_paths: list[str] = []
    split_paths = {
        "train": train_path,
        "valid": val_path,
        "test": test_path,
    }

    for split_name, path in split_paths.items():
        if path is not None and not path.exists():
            missing_paths.append(f"{split_name}: {path}")

    samples = {
        "train": _count_images(train_path) if train_path else 0,
        "valid": _count_images(val_path) if val_path else 0,
        "test": _count_images(test_path) if test_path else 0,
    }
    samples["total"] = samples["train"] + samples["valid"] + samples["test"]

    classes = config.get("names", [])
    if isinstance(classes, dict):
        classes = [classes[k] for k in sorted(classes.keys(), key=lambda x: int(x))]

    return {
        "yaml_path": str(yaml_path),
        "is_valid": len(missing_paths) == 0,
        "missing_paths": missing_paths,
        "splits": {
            "train": str(train_path) if train_path else None,
            "valid": str(val_path) if val_path else None,
            "test": str(test_path) if test_path else None,
        },
        "samples": samples,
        "num_classes": int(config.get("nc", len(classes) if classes else 0)),
        "class_names": classes,
    }
