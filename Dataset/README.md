# Dataset Folder

This folder stores dataset metadata and local split directories used by the root training pipeline.

## Expected Structure

```text
Dataset/
├── data.yaml
├── train/
│   └── images/
├── valid/
│   └── images/
└── test/
    └── images/
```

Labels should follow your YOLO format conventions in parallel label folders.

## Current data.yaml Format

`data.yaml` is expected to define:

- `path`
- `train`
- `val`
- `test` (optional but recommended)
- `nc`
- `names`

Example:

```yaml
path: .
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  - human
```

## Git Ignore Behavior

`Dataset/.gitignore` intentionally ignores bulky local splits by default:

- `train/`
- `valid/`
- `test/`

This keeps large raw datasets out of version control while preserving metadata files.

## Validate Dataset Quickly

From repository root:

```bash
./.venv/Scripts/python -c "import sys; sys.path.insert(0, 'src'); from data_loader import get_dataset_info; print(get_dataset_info())"
```
