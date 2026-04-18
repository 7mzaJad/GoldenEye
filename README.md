# GoldenEye

GoldenEye is a Search and Rescue (SAR) human-detection project focused on selecting
the best model for small/distant human detection from drone imagery.

This root project now provides a reproducible experiment pipeline to:
- train YOLO models,
- evaluate them on validation/test splits,
- compare model and dataset combinations,
- export experiment evidence (CSV, JSON, and Markdown reports).

## Install

```bash
pip install -r requirements.txt
```

## CLI Commands

All commands run from project root.

### 1) Validate Dataset Structure

```bash
python main.py dataset-info --data-yaml Dataset/data.yaml
```

### 2) Train One Model

```bash
python main.py train \
	--data-yaml Dataset/data.yaml \
	--model yolov8n.pt \
	--epochs 100 \
	--img-size 960 \
	--batch-size 16 \
	--device auto
```

### 3) Evaluate One Checkpoint

```bash
python main.py evaluate \
	--weights logs/train_runs/<run_name>/weights/best.pt \
	--data-yaml Dataset/data.yaml \
	--split test
```

### 4) Run Full Experiment Matrix (Model x Dataset)

```bash
python main.py experiment --config experiments/sar_benchmark.yaml
```

### 5) Compare and Rank Previous Runs

```bash
python main.py compare \
	--summary logs/experiments/<experiment_run>/summary.csv \
	--sort-by test_map50_95 \
	--top-k 10
```

## Experiment Config

Default matrix config:

```text
experiments/sar_benchmark.yaml
```

This file controls:
- datasets to benchmark,
- model variants,
- training hyperparameters,
- evaluation splits and thresholds,
- primary split used for ranking.

## Experiment Artifacts

Each experiment run writes to:

```text
logs/experiments/<name_timestamp>/
```

Files generated:
- `summary.csv`: flat comparison table across all runs.
- `summary.json`: machine-readable summary records.
- `detailed_runs.json`: full per-run metadata/metrics.
- `report.md`: ranked markdown report for capstone documentation.

## Recommended Selection Logic For SAR

When ranking candidates, prioritize:
- `test_map50_95` for overall detection quality,
- `test_recall` for missing fewer humans,
- qualitative false-negative review on small/distant targets.

Avoid selecting solely by speed or only by mAP50.
