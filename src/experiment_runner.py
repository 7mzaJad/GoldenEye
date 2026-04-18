from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_PATIENCE,
    DEFAULT_WORKERS,
    EXPERIMENT_RUNS_DIR,
    PROJECT_ROOT,
    resolve_path,
    timestamp_id,
)
from data_loader import get_dataset_info
from evaluate import evaluate_model
from train import train_yolo

logger = logging.getLogger(__name__)


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load experiment matrix YAML used for model x dataset benchmarking."""
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid experiment config format: {path}")

    datasets = config.get("datasets", [])
    models = config.get("models", [])
    if not datasets:
        raise ValueError("Experiment config must include at least one dataset entry.")
    if not models:
        raise ValueError("Experiment config must include at least one model entry.")

    return config


def _flatten_split_metrics(record: dict[str, Any], split_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat = dict(record)
    for split_name, metrics in split_metrics.items():
        prefix = f"{split_name}_"
        flat[f"{prefix}precision"] = metrics.get("precision")
        flat[f"{prefix}recall"] = metrics.get("recall")
        flat[f"{prefix}map50"] = metrics.get("map50")
        flat[f"{prefix}map50_95"] = metrics.get("map50_95")
        flat[f"{prefix}inference_ms"] = metrics.get("speed_inference_ms")
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _score(row: dict[str, Any], split: str) -> float:
    value = row.get(f"{split}_map50_95")
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _write_markdown_report(
    report_path: Path,
    experiment_name: str,
    description: str,
    rows: list[dict[str, Any]],
    primary_split: str,
) -> None:
    ranked = sorted(rows, key=lambda r: _score(r, primary_split), reverse=True)

    lines = [
        f"# Experiment Report: {experiment_name}",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        description or "No description provided.",
        "",
        "## Ranking",
        "",
        "| Rank | Dataset | Model | mAP50-95 | mAP50 | Recall | Precision | Train Seconds | Best Weights |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for idx, row in enumerate(ranked, start=1):
        split_prefix = f"{primary_split}_"
        lines.append(
            "| {rank} | {dataset} | {model} | {map5095:.4f} | {map50:.4f} | {recall:.4f} | {precision:.4f} | {train:.2f} | {weights} |".format(
                rank=idx,
                dataset=row.get("dataset_name", ""),
                model=row.get("model_name", ""),
                map5095=float(row.get(f"{split_prefix}map50_95") or 0.0),
                map50=float(row.get(f"{split_prefix}map50") or 0.0),
                recall=float(row.get(f"{split_prefix}recall") or 0.0),
                precision=float(row.get(f"{split_prefix}precision") or 0.0),
                train=float(row.get("train_seconds") or 0.0),
                weights=row.get("best_weights", ""),
            )
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Compare candidates primarily by mAP50-95 on your chosen target split.",
        "- For SAR drone use, inspect recall and failure cases on small/distant humans, not only overall mAP.",
        "- Keep this report with run artifacts to maintain reproducible experiment history.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment_matrix(config_path: str | Path) -> dict[str, Any]:
    """Run full model x dataset matrix and save reproducible comparison artifacts."""
    config = load_experiment_config(config_path)

    experiment_name = str(config.get("name", "sar_experiment"))
    description = str(config.get("description", "")).strip()
    run_id = f"{_sanitize_name(experiment_name)}_{timestamp_id()}"
    run_dir = EXPERIMENT_RUNS_DIR / run_id
    train_runs_dir = run_dir / "train_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_runs_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("train", {}) or {}
    eval_cfg = config.get("eval", {}) or {}
    splits = eval_cfg.get("splits", ["val", "test"])
    if not isinstance(splits, list) or not splits:
        splits = ["val", "test"]
    primary_split = str(eval_cfg.get("primary_split", splits[0]))

    rows: list[dict[str, Any]] = []
    detailed_runs: list[dict[str, Any]] = []

    for dataset in config.get("datasets", []):
        dataset_name = str(dataset.get("name", "dataset"))
        data_yaml = str(dataset.get("data_yaml"))
        if not data_yaml:
            logger.warning("Skipping dataset entry with missing data_yaml: %s", dataset)
            continue

        data_yaml_path = resolve_path(data_yaml, base_dir=PROJECT_ROOT)
        dataset_info = get_dataset_info(data_yaml_path)
        if not dataset_info.get("is_valid", False):
            logger.warning(
                "Dataset invalid, skipping: %s missing=%s",
                data_yaml_path,
                dataset_info.get("missing_paths", []),
            )
            continue

        for model_entry in config.get("models", []):
            model_name = str(model_entry.get("name", "model"))
            model_weights = str(model_entry.get("weights", DEFAULT_MODEL_WEIGHTS))

            run_tag = _sanitize_name(f"{model_name}_{dataset_name}_{timestamp_id()}")
            logger.info(
                "Experiment run started: dataset=%s model=%s run=%s",
                dataset_name,
                model_name,
                run_tag,
            )

            train_output = train_yolo(
                data_yaml=str(data_yaml_path),
                model_weights=model_weights,
                epochs=int(train_cfg.get("epochs", DEFAULT_EPOCHS)),
                batch_size=int(train_cfg.get("batch_size", DEFAULT_BATCH_SIZE)),
                img_size=int(train_cfg.get("img_size", DEFAULT_IMG_SIZE)),
                device=str(train_cfg.get("device", DEFAULT_DEVICE)),
                run_name=run_tag,
                project_dir=train_runs_dir,
                patience=int(train_cfg.get("patience", DEFAULT_PATIENCE)),
                workers=int(train_cfg.get("workers", DEFAULT_WORKERS)),
                exist_ok=bool(train_cfg.get("exist_ok", False)),
                extra_train_args=dict(train_cfg.get("extra_train_args", {}) or {}),
            )

            weights_for_eval = (
                train_output.get("best_weights")
                or train_output.get("last_weights")
                or str(model_weights)
            )

            split_metrics: dict[str, dict[str, Any]] = {}
            for split in splits:
                split_metrics[split] = evaluate_model(
                    weights_path=weights_for_eval,
                    data_yaml=str(data_yaml_path),
                    split=str(split),
                    img_size=int(eval_cfg.get("img_size", train_cfg.get("img_size", DEFAULT_IMG_SIZE))),
                    batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", DEFAULT_BATCH_SIZE))),
                    device=str(eval_cfg.get("device", train_cfg.get("device", DEFAULT_DEVICE))),
                    conf=float(eval_cfg.get("conf", DEFAULT_CONF)),
                    iou=float(eval_cfg.get("iou", DEFAULT_IOU)),
                    max_det=int(eval_cfg.get("max_det", DEFAULT_MAX_DET)),
                    extra_eval_args=dict(eval_cfg.get("extra_eval_args", {}) or {}),
                )

            base_record = {
                "experiment_name": experiment_name,
                "run_id": run_id,
                "dataset_name": dataset_name,
                "dataset_yaml": str(data_yaml_path),
                "model_name": model_name,
                "model_weights": model_weights,
                "best_weights": train_output.get("best_weights"),
                "train_run_dir": train_output.get("save_dir"),
                "train_seconds": train_output.get("train_seconds"),
                "samples_train": dataset_info["samples"]["train"],
                "samples_valid": dataset_info["samples"]["valid"],
                "samples_test": dataset_info["samples"]["test"],
                "num_classes": dataset_info["num_classes"],
            }

            flat_record = _flatten_split_metrics(base_record, split_metrics)
            rows.append(flat_record)

            detailed_runs.append(
                {
                    "record": base_record,
                    "train_output": train_output,
                    "dataset_info": dataset_info,
                    "split_metrics": split_metrics,
                }
            )

            logger.info(
                "Experiment run finished: dataset=%s model=%s %s_map50_95=%s",
                dataset_name,
                model_name,
                primary_split,
                flat_record.get(f"{primary_split}_map50_95"),
            )

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    detailed_json = run_dir / "detailed_runs.json"
    report_md = run_dir / "report.md"

    _write_csv(summary_csv, rows)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(detailed_json, "w", encoding="utf-8") as f:
        json.dump(detailed_runs, f, indent=2)

    _write_markdown_report(
        report_path=report_md,
        experiment_name=experiment_name,
        description=description,
        rows=rows,
        primary_split=primary_split,
    )

    result = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "detailed_json": str(detailed_json),
        "report_md": str(report_md),
        "num_runs": len(rows),
        "primary_split": primary_split,
    }
    logger.info("Experiment matrix completed: %s", result)
    return result


def compare_summary_file(
    summary_csv: str | Path,
    sort_by: str = "test_map50_95",
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Load summary CSV and return top-k ranked rows by selected metric."""
    path = resolve_path(summary_csv)
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def key_fn(row: dict[str, Any]) -> float:
        try:
            return float(row.get(sort_by, "-1") or -1)
        except (TypeError, ValueError):
            return -1.0

    ranked = sorted(rows, key=key_fn, reverse=True)
    return ranked[:max(int(top_k), 1)]
