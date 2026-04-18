"""GoldenEye CLI for reproducible SAR object-detection experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONF,
    DEFAULT_DATA_YAML,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_PATIENCE,
    DEFAULT_WORKERS,
    EXPERIMENTS_DIR,
)
from data_loader import get_dataset_info  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from experiment_runner import compare_summary_file, run_experiment_matrix  # noqa: E402
from train import train_yolo  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _dump(data: dict) -> None:
    print(json.dumps(data, indent=2))


def cmd_dataset_info(args: argparse.Namespace) -> int:
    info = get_dataset_info(args.data_yaml)
    _dump(info)
    return 0 if info.get("is_valid") else 1


def cmd_train(args: argparse.Namespace) -> int:
    result = train_yolo(
        data_yaml=args.data_yaml,
        model_weights=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        run_name=args.run_name,
        project_dir=args.project,
        patience=args.patience,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )
    _dump(result)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_model(
        weights_path=args.weights,
        data_yaml=args.data_yaml,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    _dump(result)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    result = run_experiment_matrix(args.config)
    _dump(result)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    top_rows = compare_summary_file(
        summary_csv=args.summary,
        sort_by=args.sort_by,
        top_k=args.top_k,
    )
    print(json.dumps(top_rows, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GoldenEye SAR training/evaluation pipeline",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # dataset-info
    p_info = subparsers.add_parser("dataset-info", help="Validate dataset and print sample stats")
    p_info.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML))
    p_info.set_defaults(func=cmd_dataset_info)

    # train
    p_train = subparsers.add_parser("train", help="Train one model on one dataset")
    p_train.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML))
    p_train.add_argument("--model", default=DEFAULT_MODEL_WEIGHTS)
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p_train.add_argument("--device", default=DEFAULT_DEVICE)
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--project", default="logs/train_runs")
    p_train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p_train.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_train.add_argument("--exist-ok", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model checkpoint")
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML))
    p_eval.add_argument("--split", default="val")
    p_eval.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p_eval.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_eval.add_argument("--device", default=DEFAULT_DEVICE)
    p_eval.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p_eval.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p_eval.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    p_eval.set_defaults(func=cmd_evaluate)

    # experiment
    p_exp = subparsers.add_parser("experiment", help="Run full model x dataset matrix")
    p_exp.add_argument(
        "--config",
        default=str(EXPERIMENTS_DIR / "sar_benchmark.yaml"),
        help="Path to experiment matrix YAML",
    )
    p_exp.set_defaults(func=cmd_experiment)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Rank runs from a summary CSV")
    p_cmp.add_argument("--summary", required=True, help="Path to summary.csv")
    p_cmp.add_argument("--sort-by", default="test_map50_95")
    p_cmp.add_argument("--top-k", type=int, default=10)
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
