"""Dataset inspection helpers for the root YOLO pipeline."""

from pathlib import Path
from typing import Any

import yaml

from config import DATA_YAML, IMAGE_EXTENSIONS


def _resolve_split_path(raw_value: Any, base_dir: Path) -> Path:
    split_path = Path(str(raw_value)).expanduser()
    if not split_path.is_absolute():
        split_path = (base_dir / split_path).resolve()
    return split_path


def _count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_dataset_info(data_yaml: str = DATA_YAML) -> dict[str, Any]:
    """Validate dataset YAML and return split/sample metadata."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    info: dict[str, Any] = {
        "is_valid": False,
        "yaml_path": str(yaml_path),
        "paths": {"train": None, "valid": None, "test": None},
        "samples": {"train": 0, "valid": 0, "test": 0, "total": 0},
        "errors": [],
    }

    if not yaml_path.exists():
        info["errors"].append(f"Dataset config not found: {yaml_path}")
        return info

    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        info["errors"].append(f"Failed to parse dataset YAML: {exc}")
        return info

    if not isinstance(cfg, dict):
        info["errors"].append("Dataset YAML root must be a dictionary.")
        return info

    yaml_dir = yaml_path.parent
    raw_dataset_root = cfg.get("path", ".")
    dataset_root = _resolve_split_path(raw_dataset_root, yaml_dir)
    if not dataset_root.exists():
        info["errors"].append(
            f"Dataset root directory from 'path' does not exist: {dataset_root}"
        )

    split_key_map = {"train": "train", "val": "valid", "test": "test"}

    for yaml_key, out_key in split_key_map.items():
        raw = cfg.get(yaml_key)
        if raw is None:
            if yaml_key != "test":
                info["errors"].append(f"Missing required split in YAML: {yaml_key}")
            continue

        resolved = _resolve_split_path(raw, dataset_root)
        info["paths"][out_key] = str(resolved)
        info["samples"][out_key] = _count_images(resolved)

        if yaml_key in {"train", "val"} and not resolved.exists():
            info["errors"].append(f"Missing split directory: {resolved}")

    names = cfg.get("names")
    nc = cfg.get("nc")
    if not isinstance(names, list) or not names:
        info["errors"].append("Dataset YAML must define a non-empty 'names' list.")
    if not isinstance(nc, int) or nc <= 0:
        info["errors"].append("Dataset YAML must define a positive integer 'nc'.")

    info["samples"]["total"] = (
        info["samples"]["train"] + info["samples"]["valid"] + info["samples"]["test"]
    )
    info["is_valid"] = len(info["errors"]) == 0
    return info
