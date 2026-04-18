"""Lightweight file preprocessing helpers for batch inference."""

from pathlib import Path

from config import IMAGE_EXTENSIONS


def list_image_files(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted image files under a folder."""
    root = Path(folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
