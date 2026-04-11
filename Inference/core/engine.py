"""
ONNX Runtime inference engine.

Handles:
- Session creation with optimized execution providers
- Graph optimization for production performance
- Warm-up inference to avoid first-hit latency
- Thread-safe predict() method
- Proper resource cleanup
"""

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from core.preprocessing import preprocess, PreprocessResult
from core.postprocessing import postprocess
from utils.drawing import Detection

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages an ONNX inference session and runs predictions.

    Usage:
        engine = InferenceEngine("model.onnx", "yolov8", ["human"])
        engine.load()
        detections, latency = engine.predict(frame, conf=0.25, iou=0.45)
    """

    def __init__(
        self,
        model_path: str | Path,
        model_type: str,
        class_names: list[str],
        input_size: int = 640,
    ):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.class_names = class_names
        self.input_size = input_size

        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Load the ONNX model and run warm-up inference.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError:      If ONNX Runtime fails to create session.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path.resolve()}"
            )

        logger.info(f"Loading ONNX model: {self.model_path.name}")

        # Session options for production performance
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True

        # Execution providers — prefer CoreML on macOS, fallback to CPU
        providers = self._get_providers()

        try:
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=providers
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model: {e}\n"
                f"Path: {self.model_path.resolve()}"
            ) from e

        self._input_name = self._session.get_inputs()[0].name
        self._is_loaded = True

        # Log session info
        active_provider = self._session.get_providers()[0]
        logger.info(f"Model loaded successfully on {active_provider}")
        logger.info(f"Input: {self._input_name}, Size: {self.input_size}×{self.input_size}")

        # Warm-up: run 2 dummy inferences to pre-allocate memory
        self._warmup()

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> tuple[list[Detection], float]:
        """Run inference on a single BGR frame.

        Args:
            frame:          BGR image from OpenCV.
            conf_threshold: Confidence threshold for filtering.
            iou_threshold:  IoU threshold for NMS (YOLOv8 only).

        Returns:
            Tuple of (list of Detections, inference_time_ms).
        """
        if not self._is_loaded or self._session is None:
            return [], 0.0

        # Preprocess
        prep = preprocess(frame, self.input_size)

        # Inference (timed)
        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: prep.tensor})
        inference_ms = (time.perf_counter() - t0) * 1000

        # Postprocess
        detections = postprocess(
            output=outputs[0],
            preprocess_info=prep,
            class_names=self.class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_type=self.model_type,
        )

        return detections, inference_ms

    def unload(self) -> None:
        """Release the inference session."""
        self._session = None
        self._is_loaded = False
        logger.info("Inference session unloaded")

    def _warmup(self) -> None:
        """Run dummy inferences to pre-allocate ONNX Runtime buffers."""
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(2):
            self._session.run(None, {self._input_name: dummy})
        logger.info("Warm-up inference completed")

    @staticmethod
    def _get_providers() -> list[str]:
        """Return available execution providers in priority order."""
        available = ort.get_available_providers()
        preferred = [
            "CoreMLExecutionProvider",    # macOS Apple Silicon
            "CUDAExecutionProvider",      # NVIDIA GPU
            "CPUExecutionProvider",       # Fallback
        ]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]
