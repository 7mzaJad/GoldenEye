"""
Inference engine with deployable routing between:
- Standard ONNX full-frame inference
- Optional SAHI sliced inference

Handles:
- Session creation and warm-up for the existing ONNX path
- Optional SAHI initialization using the same trained model weights
- Mode-aware routing with safe fallback to standard inference
- Unified Detection output format for stable downstream rendering/UI
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from core.preprocessing import preprocess, PreprocessResult
from core.postprocessing import postprocess
from core.sahi_engine import SahiInferenceEngine
from utils.drawing import Detection

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages inference sessions and routes requests by deployment mode.

    Usage:
        engine = InferenceEngine("model.onnx", "yolov8", ["human"])
        engine.load()
        detections, latency = engine.predict(frame, conf_threshold=0.25, iou_threshold=0.45)
    """

    def __init__(
        self,
        model_path: str | Path,
        model_type: str,
        class_names: list[str],
        input_size: int = 640,
        sahi_model_path: str | Path | None = None,
        sahi_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
    ):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.class_names = class_names
        self.input_size = input_size
        self.sahi_model_path = Path(sahi_model_path) if sahi_model_path else None

        self._sahi_config = dict(sahi_config or {})
        self._inference_config = dict(inference_config or {})

        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._is_loaded = False

        # Inference mode is request-driven and can be switched at runtime.
        self._inference_mode = str(self._inference_config.get("mode", "standard")).lower()

        # SAHI is inference-only and optional; failures should not kill standard mode.
        self._sahi_engine: SahiInferenceEngine | None = None
        self._sahi_available = False

        # Live-source controls protect real-time UX when SAHI is too slow per frame.
        self._enable_live_sahi = bool(
            self._inference_config.get("enable_sahi_for_live_sources", False)
        )
        self._live_sahi_every_n_frames = max(
            1, int(self._inference_config.get("sahi_live_every_n_frames", 5))
        )
        self._live_frame_counter = 0
        self._fallback_to_standard = bool(
            self._inference_config.get("fallback_to_standard_on_error", True)
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def inference_mode(self) -> str:
        return self._inference_mode

    def set_mode(self, mode: str) -> None:
        """Set requested inference mode: 'standard' or 'sahi'."""
        normalized = str(mode).lower().strip()
        self._inference_mode = normalized if normalized in {"standard", "sahi"} else "standard"
        self._live_frame_counter = 0

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

        # SAHI init is best-effort: if it fails, standard inference remains available.
        self._initialize_sahi()

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        mode: str | None = None,
        source_type: str = "unknown",
        return_meta: bool = False,
    ) -> tuple[list[Detection], float] | tuple[list[Detection], float, dict[str, Any]]:
        """Run inference on a single BGR frame.

        Args:
            frame:          BGR image from OpenCV.
            conf_threshold: Confidence threshold for filtering.
            iou_threshold:  IoU threshold for NMS (YOLOv8 only).
            mode:           Optional override for current request ('standard' or 'sahi').
            source_type:    Source category ('image', 'camera', 'screen', 'video').
            return_meta:    If True, include routing/performance metadata.

        Returns:
            Tuple of (detections, inference_time_ms[, metadata]).
        """
        requested_mode = str(mode or self._inference_mode).lower().strip()
        if requested_mode not in {"standard", "sahi"}:
            requested_mode = "standard"

        height, width = frame.shape[:2]
        meta: dict[str, Any] = {
            "requested_mode": requested_mode,
            "actual_mode": "standard",
            "source_type": source_type,
            "resolution": {"width": int(width), "height": int(height)},
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "confidence_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "latency_ms": 0.0,
            "num_detections": 0,
            "fallback_reason": None,
            "sahi": None,
        }

        if not self._is_loaded or self._session is None:
            if return_meta:
                return [], 0.0, meta
            return [], 0.0

        detections: list[Detection]
        inference_ms: float

        if requested_mode == "sahi":
            should_run_sahi, fallback_reason = self._should_run_sahi(source_type)
            if should_run_sahi and self._sahi_engine is not None:
                try:
                    t0 = time.perf_counter()
                    detections = self._sahi_engine.predict(frame, conf_threshold=conf_threshold)
                    inference_ms = (time.perf_counter() - t0) * 1000.0
                    meta["actual_mode"] = "sahi"
                    meta["model_path"] = str(self.sahi_model_path) if self.sahi_model_path else str(self.model_path)
                    meta["sahi"] = self._sahi_engine.settings_for_logs()
                except Exception as exc:
                    logger.exception("SAHI inference failed: %s", exc)
                    if self._fallback_to_standard:
                        detections, inference_ms = self._predict_standard(
                            frame,
                            conf_threshold,
                            iou_threshold,
                        )
                        meta["fallback_reason"] = "sahi_runtime_error"
                    else:
                        raise
            else:
                detections, inference_ms = self._predict_standard(
                    frame,
                    conf_threshold,
                    iou_threshold,
                )
                meta["fallback_reason"] = fallback_reason
        else:
            detections, inference_ms = self._predict_standard(
                frame,
                conf_threshold,
                iou_threshold,
            )

        meta["num_detections"] = len(detections)
        meta["latency_ms"] = float(inference_ms)
        if return_meta:
            return detections, inference_ms, meta
        return detections, inference_ms

    def unload(self) -> None:
        """Release the inference session."""
        self._session = None
        if self._sahi_engine is not None:
            self._sahi_engine.unload()
        self._sahi_engine = None
        self._sahi_available = False
        self._is_loaded = False
        logger.info("Inference session unloaded")

    def _warmup(self) -> None:
        """Run dummy inferences to pre-allocate ONNX Runtime buffers."""
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(2):
            self._session.run(None, {self._input_name: dummy})
        logger.info("Warm-up inference completed")

    def _initialize_sahi(self) -> None:
        """Initialize SAHI once and keep it ready for optional routed use."""
        self._sahi_available = False

        if not bool(self._sahi_config.get("enabled", True)):
            logger.info("SAHI is disabled in config; standard mode remains available")
            return

        if self.sahi_model_path is None:
            logger.warning("SAHI model path is not configured; SAHI mode will fallback")
            return

        if not self.sahi_model_path.exists():
            logger.warning(
                "SAHI model path does not exist (%s); SAHI mode will fallback",
                self.sahi_model_path,
            )
            return

        settings = dict(self._sahi_config)
        settings.setdefault("confidence_threshold", self._inference_config.get("confidence_threshold", 0.25))

        try:
            self._sahi_engine = SahiInferenceEngine(
                model_path=self.sahi_model_path,
                class_names=self.class_names,
                settings=settings,
            )
            self._sahi_engine.load()
            self._sahi_available = True
            logger.info("SAHI initialized and ready")
        except Exception as exc:
            self._sahi_engine = None
            self._sahi_available = False
            logger.warning("SAHI initialization failed, keeping standard mode only: %s", exc)

    def _should_run_sahi(self, source_type: str) -> tuple[bool, str | None]:
        """Decide whether SAHI should run for this request.

        This keeps deployment practical by allowing slower SAHI mode to be throttled
        on live sources while still fully available for still-image inspection.
        """
        if not self._sahi_available or self._sahi_engine is None:
            return False, "sahi_unavailable"

        live_sources = {"camera", "screen", "video"}
        if source_type in live_sources:
            if not self._enable_live_sahi:
                return False, "sahi_disabled_for_live_sources"

            self._live_frame_counter += 1
            if self._live_sahi_every_n_frames > 1:
                if (self._live_frame_counter % self._live_sahi_every_n_frames) != 0:
                    return False, "sahi_throttled_live_frame"

        return True, None

    def _predict_standard(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> tuple[list[Detection], float]:
        """Run the unchanged existing ONNX inference path."""
        prep: PreprocessResult = preprocess(frame, self.input_size)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: prep.tensor})
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = postprocess(
            output=outputs[0],
            preprocess_info=prep,
            class_names=self.class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_type=self.model_type,
        )

        return detections, inference_ms

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
