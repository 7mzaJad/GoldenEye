"""
Main application window — orchestrates all panels, camera thread,
and inference engine into a cohesive real-time SAR human detection interface.
"""

import logging
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStatusBar, QMessageBox,
)

from core.engine import InferenceEngine
from handlers.camera import CameraThread, scan_cameras, get_preferred_camera
from handlers.screen import ScreenThread, scan_monitors
from handlers.image import ImageHandler
from utils.buffer import FrameBuffer
from utils.config_loader import load_config, resolve_model_path, resolve_sahi_model_path
from utils.drawing import draw_detections, Detection
from utils.constants import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT

from ui.live_view_panel import LiveViewPanel
from ui.upload_panel import UploadPanel
from ui.controls_panel import ControlsPanel
from ui.results_panel import ResultsPanel
from ui.styles import MAIN_STYLESHEET

logger = logging.getLogger(__name__)


# ── Inference Worker (runs in a separate thread) ──────────────────────────

class InferenceWorker(QObject):
    """Runs ONNX inference in a background thread.

    Receives frames via process_frame(), runs the engine,
    and emits results back to the main thread.
    """

    result_ready = Signal(np.ndarray, list, float, dict)  # annotated_frame, detections, ms, meta
    error = Signal(str)

    def __init__(self, buffer: FrameBuffer):
        super().__init__()
        self._buffer = buffer
        self._engine: InferenceEngine | None = None
        self._conf_threshold = 0.25
        self._iou_threshold = 0.45
        self._inference_mode = "standard"
        self._source_type = "unknown"
        self._busy = False

    def set_engine(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def set_thresholds(self, conf: float, iou: float) -> None:
        self._conf_threshold = conf
        self._iou_threshold = iou

    def set_inference_mode(self, mode: str) -> None:
        self._inference_mode = str(mode).lower().strip()

    def set_source_type(self, source_type: str) -> None:
        self._source_type = source_type

    @property
    def is_busy(self) -> bool:
        return self._busy

    @Slot()
    def trigger_inference(self) -> None:
        """Run inference on the newest frame from the buffer, dropping intermediate ones."""
        if self._busy or self._engine is None or not self._engine.is_loaded:
            return

        # Fetch latest frame, which clears the buffer. If None, we've already processed it.
        frame = self._buffer.get()
        if frame is None:
            return

        self._busy = True
        try:
            detections, inference_ms, meta = self._engine.predict(
                frame,
                self._conf_threshold,
                self._iou_threshold,
                mode=self._inference_mode,
                source_type=self._source_type,
                return_meta=True,
            )
            annotated = frame.copy()
            draw_detections(annotated, detections)
            self.result_ready.emit(annotated, detections, inference_ms, meta)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._busy = False


# ── Main Window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """GoldenEye main application window."""

    # Internal signal to trigger the inference worker thread
    _trigger_inference = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GoldenEye — Real-Time Search and Rescue Human Detection")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # State
        self._config = load_config()
        self._inference_cfg = self._config.get("inference", {})
        self._engine: InferenceEngine | None = None
        self._camera_thread: CameraThread | None = None
        self._screen_thread: ScreenThread | None = None
        self._image_handler: ImageHandler | None = None
        self._fps_times: deque[float] = deque(maxlen=30)
        self._stream_log_counter = 0
        self._requested_mode = str(self._inference_cfg.get("mode", "standard")).lower()
        
        self._cameras: list = []           
        self._monitors: list = []
        self._current_source: str = "camera"
        self._selected_cam_index: int = 0
        self._selected_monitor_index: int = 0
        
        self._frame_buffer = FrameBuffer() # Single-slot overwrite buffer for zero latency

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        # Auto-scan connected cameras and monitors
        self._scan_cameras()
        self._scan_monitors()

        # Build UI
        self._setup_ui()

        # Setup inference worker thread
        self._setup_worker()

        # Load default model
        default_key = self._config["defaults"].get("default_model", "yolov8n")
        QTimer.singleShot(100, lambda: self._load_model(default_key))

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # ── Left: Controls Panel ───────────────────────────────────────
        model_keys = list(self._config["models"].keys())
        self._controls = ControlsPanel(
            model_keys,
            self._config["defaults"],
            devices=self._cameras,
            inference_defaults=self._inference_cfg,
        )
        self._controls.setFixedWidth(280)
        self._controls.start_capture.connect(self._start_capture)
        self._controls.stop_capture.connect(self._stop_capture)
        self._controls.model_changed.connect(self._load_model)
        self._controls.device_changed.connect(self._on_device_changed)
        self._controls.source_changed.connect(self._on_source_changed)
        self._controls.inference_mode_changed.connect(self._on_inference_mode_changed)
        self._controls.conf_threshold_changed.connect(self._on_conf_changed)
        self._controls.iou_threshold_changed.connect(self._on_iou_changed)
        self._controls.refresh_button.clicked.connect(self._rescan_devices)

        # Auto-select the preferred camera in the dropdown initially
        self._controls.set_devices(self._cameras)
        self._controls.select_device_index(self._selected_cam_index)
        self._controls.set_inference_mode(self._requested_mode)

        root_layout.addWidget(self._controls)

        # ── Center: Camera + Upload ────────────────────────────────────
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)

        self._live_panel = LiveViewPanel()
        center_layout.addWidget(self._live_panel, stretch=3)

        self._upload_panel = UploadPanel()
        self._upload_panel.image_selected.connect(self._on_image_selected)
        center_layout.addWidget(self._upload_panel, stretch=2)

        root_layout.addLayout(center_layout, stretch=1)

        # ── Right: Results Panel ───────────────────────────────────────
        self._results = ResultsPanel()
        self._results.setFixedWidth(300)
        root_layout.addWidget(self._results)

        # ── Status Bar ─────────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready — Select a model and start detection")

    def _setup_worker(self) -> None:
        """Create the inference worker and move it to a background thread."""
        self._worker_thread = QThread()
        self._worker = InferenceWorker(self._frame_buffer)
        self._worker.moveToThread(self._worker_thread)

        # Connect signals
        self._trigger_inference.connect(self._worker.trigger_inference)
        self._worker.result_ready.connect(self._on_inference_result)
        self._worker.error.connect(self._on_inference_error)

        self._worker_thread.start()
        self._worker.set_thresholds(
            self._controls.confidence_threshold,
            self._controls.iou_threshold,
        )
        self._worker.set_inference_mode(self._requested_mode)
        self._worker.set_source_type(self._current_source)

    # ── Model Loading ──────────────────────────────────────────────────

    def _load_model(self, model_key: str) -> None:
        """Load an ONNX model by its config key."""
        self._controls.set_model_loading()
        self._statusbar.showMessage(f"Loading model: {model_key}…")

        # Stop capture while switching models
        self._stop_capture()

        # Unload previous engine
        if self._engine:
            self._engine.unload()

        try:
            model_cfg = self._config["models"][model_key]
            model_path = resolve_model_path(self._config, model_key)
            sahi_model_path = resolve_sahi_model_path(self._config, model_key)

            self._engine = InferenceEngine(
                model_path=model_path,
                model_type=model_cfg["type"],
                class_names=self._config["class_names"],
                input_size=self._config["defaults"].get("input_size", 640),
                sahi_model_path=sahi_model_path,
                sahi_config=self._config.get("sahi", {}),
                inference_config=self._inference_cfg,
            )
            self._engine.load()
            self._engine.set_mode(self._requested_mode)

            # Update worker
            self._worker.set_engine(self._engine)
            self._worker.set_inference_mode(self._requested_mode)
            self._image_handler = ImageHandler(self._engine)

            self._controls.set_model_loaded(True, model_key.upper())
            self._statusbar.showMessage(
                f"Model loaded: {model_key.upper()} — Mode: {self._requested_mode.upper()}"
            )

        except (FileNotFoundError, RuntimeError) as e:
            self._controls.set_model_loaded(False)
            self._statusbar.showMessage("Model loading failed")
            QMessageBox.critical(
                self, "Model Error",
                f"Failed to load model '{model_key}':\n\n{e}"
            )

    # ── Capture Controls ───────────────────────────────────────────────

    def _start_capture(self) -> None:
        if not self._engine or not self._engine.is_loaded:
            QMessageBox.warning(
                self, "No Model",
                "Please load a model before starting capture."
            )
            return

        idx = self._controls.selected_device_index
        if idx < 0:
            QMessageBox.warning(
                self, "No Device",
                "No device selected.\nClick 'Rescan Devices' to detect sources."
            )
            return

        self._frame_buffer.clear()
        self._worker.set_source_type(self._current_source)
        
        dev_name = "Unknown"
        if self._current_source == "camera":
            self._camera_thread = CameraThread(idx, self._frame_buffer)
            self._camera_thread.frame_ready.connect(self._on_live_frame)
            self._camera_thread.error.connect(self._on_capture_error)
            self._camera_thread.start()
            
            dev_name = f"Camera {idx}"
            for dev in self._cameras:
                if dev.index == idx: dev_name = dev.name; break
        else:
            mon_dict = {}
            dev_name = f"Display {idx}"
            for dev in self._monitors:
                if dev.index == idx: 
                    dev_name = dev.name
                    mon_dict = dev.mss_monitor_dict
                    break
            
            self._screen_thread = ScreenThread(mon_dict, self._frame_buffer)
            self._screen_thread.frame_ready.connect(self._on_live_frame)
            self._screen_thread.error.connect(self._on_capture_error)
            self._screen_thread.start()

        self._fps_times.clear()
        self._stream_log_counter = 0
        self._controls.set_capture_running(True)
        self._controls.set_inference_state("running")

        self._statusbar.showMessage(
            f"Capture active — {dev_name} — Running SARD detection"
        )

    def _stop_capture(self) -> None:
        if self._camera_thread:
            self._camera_thread.frame_ready.disconnect(self._on_live_frame)
            self._camera_thread.stop()
            self._camera_thread = None
            
        if self._screen_thread:
            self._screen_thread.frame_ready.disconnect(self._on_live_frame)
            self._screen_thread.stop()
            self._screen_thread = None

        self._controls.set_capture_running(False)
        self._controls.set_inference_state("idle")
        self._live_panel.show_placeholder()
        self._results.clear()
        self._statusbar.showMessage("Capture stopped")

    # ── Signal Handlers ────────────────────────────────────────────────

    @Slot()
    def _on_live_frame(self) -> None:
        """Capture source wrote a new frame properly — trigger worker if idle."""
        self._trigger_inference.emit()

    @Slot(np.ndarray, list, float, dict)
    def _on_inference_result(
        self,
        annotated: np.ndarray,
        detections: list,
        inference_ms: float,
        meta: dict,
    ) -> None:
        """Received inference results — update display."""
        # FPS calculation
        now = time.perf_counter()
        self._fps_times.append(now)
        fps = None
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = (len(self._fps_times) - 1) / elapsed

        self._live_panel.update_frame(annotated)
        self._results.update_results(detections, inference_ms, fps)
        self._log_inference_metrics(meta, fps)

    @Slot(str)
    def _on_inference_error(self, message: str) -> None:
        logger.error(f"Inference error: {message}")
        self._controls.set_inference_state("error")
        self._statusbar.showMessage(f"Inference error: {message}")

    @Slot(str)
    def _on_capture_error(self, message: str) -> None:
        self._controls.set_capture_running(False)
        self._controls.set_inference_state("idle")
        self._live_panel.show_error(message)
        QMessageBox.warning(self, "Capture Error", message)

    @Slot(str)
    def _on_image_selected(self, file_path: str) -> None:
        """Run prediction on an uploaded image."""
        if not self._image_handler:
            QMessageBox.warning(
                self, "No Model",
                "Please load a model before uploading an image."
            )
            return

        self._upload_panel.show_loading()

        try:
            conf = self._controls.confidence_threshold
            iou = self._controls.iou_threshold
            _, annotated, detections, inference_ms, meta = self._image_handler.predict(
                file_path,
                conf,
                iou,
                inference_mode=self._requested_mode,
            )
            self._upload_panel.show_result(annotated, len(detections), inference_ms)
            self._log_inference_metrics(meta, fps=None, force=True)

            # Also update results panel if camera is not running
            live_running = (
                (self._camera_thread and self._camera_thread.is_running)
                or (self._screen_thread and self._screen_thread.is_running)
            )
            if not live_running:
                self._results.update_results(detections, inference_ms)
                self._statusbar.showMessage(
                    f"Image prediction: {len(detections)} detections "
                    f"in {inference_ms:.1f}ms"
                )

        except ValueError as e:
            self._upload_panel.show_error(str(e))
        except Exception as e:
            self._upload_panel.show_error(f"Prediction failed: {e}")
            logger.exception("Image prediction error")

    def _on_conf_changed(self, value: float) -> None:
        self._worker.set_thresholds(value, self._controls.iou_threshold)

    def _on_iou_changed(self, value: float) -> None:
        self._worker.set_thresholds(self._controls.confidence_threshold, value)

    def _on_source_changed(self, source: str) -> None:
        """User switched between Camera and Screen capture modes."""
        self._current_source = source
        self._worker.set_source_type(source)
        if source == "camera":
            self._controls.set_devices(self._cameras)
            self._controls.select_device_index(self._selected_cam_index)
        else:
            self._controls.set_devices(self._monitors)
            self._controls.select_device_index(self._selected_monitor_index)

    def _on_inference_mode_changed(self, mode: str) -> None:
        """User switched runtime inference mode between standard and SAHI."""
        self._requested_mode = str(mode).lower().strip()

        if self._engine:
            self._engine.set_mode(self._requested_mode)
        self._worker.set_inference_mode(self._requested_mode)

        self._statusbar.showMessage(
            f"Inference mode set to {self._requested_mode.upper()}"
        )

    def _on_device_changed(self, index: int) -> None:
        """User picked a different device from the dropdown."""
        if self._current_source == "camera":
            self._selected_cam_index = index
        else:
            self._selected_monitor_index = index
        logger.info(f"{self._current_source.capitalize()} selection changed to index {index}")

    def _log_inference_metrics(
        self,
        meta: dict[str, Any],
        fps: float | None,
        force: bool = False,
    ) -> None:
        """Log mode-aware inference metrics without changing prediction behavior.

        For live streams we throttle logs to avoid excessive log volume.
        """
        source_type = str(meta.get("source_type", "unknown"))
        is_live = source_type in {"camera", "screen", "video"}

        if is_live and not force:
            self._stream_log_counter += 1
            if self._stream_log_counter % 15 != 0:
                return

        resolution = meta.get("resolution", {})
        width = resolution.get("width", "?")
        height = resolution.get("height", "?")

        logger.info(
            "Inference metrics: requested_mode=%s actual_mode=%s source=%s latency_ms=%.2f "
            "fps=%s detections=%s resolution=%sx%s conf=%.2f iou=%.2f model=%s",
            meta.get("requested_mode"),
            meta.get("actual_mode"),
            source_type,
            float(meta.get("latency_ms", 0.0)),
            f"{fps:.2f}" if fps is not None else "n/a",
            meta.get("num_detections", 0),
            width,
            height,
            float(meta.get("confidence_threshold", 0.0)),
            float(meta.get("iou_threshold", 0.0)),
            meta.get("model_path", ""),
        )

        fallback_reason = meta.get("fallback_reason")
        if fallback_reason:
            logger.info(
                "Inference fallback: requested_mode=%s actual_mode=%s reason=%s",
                meta.get("requested_mode"),
                meta.get("actual_mode"),
                fallback_reason,
            )

        if meta.get("actual_mode") == "sahi":
            sahi = meta.get("sahi") or {}
            logger.info(
                "SAHI settings: slice_height=%s slice_width=%s overlap_h=%.3f overlap_w=%.3f "
                "postprocess_type=%s postprocess_metric=%s postprocess_threshold=%.3f",
                sahi.get("slice_height"),
                sahi.get("slice_width"),
                float(sahi.get("overlap_height_ratio", 0.0)),
                float(sahi.get("overlap_width_ratio", 0.0)),
                sahi.get("postprocess_type"),
                sahi.get("postprocess_match_metric"),
                float(sahi.get("postprocess_match_threshold", 0.0)),
            )

    # ── Camera Scanning ─────────────────────────────────────────────

    def _scan_cameras(self) -> None:
        """Probe camera devices and pick the preferred one."""
        max_index = self._config["defaults"].get("camera_scan_max", 5)
        fallback = self._config["defaults"].get("camera_index", 0)

        self._cameras = scan_cameras(max_index=max_index)
        self._selected_cam_index = get_preferred_camera(
            self._cameras, fallback_index=fallback
        )

        if self._cameras:
            names = ", ".join(d.display_name for d in self._cameras)
            logger.info(f"Detected cameras: {names}")
            logger.info(f"Auto-selected camera index: {self._selected_cam_index}")
        else:
            logger.warning("No cameras found during scan")

    def _scan_monitors(self) -> None:
        """Probe available display screens."""
        self._monitors = scan_monitors()
        if self._monitors:
            self._selected_monitor_index = self._monitors[0].index

    def _rescan_devices(self) -> None:
        """Re-scan current source devices and refresh the dropdown."""
        self._statusbar.showMessage("Scanning devices…")
        
        if self._current_source == "camera":
            self._scan_cameras()
            self._controls.set_devices(self._cameras)
            self._controls.select_device_index(self._selected_cam_index)
            count = len(self._cameras)
        else:
            self._scan_monitors()
            self._controls.set_devices(self._monitors)
            self._controls.select_device_index(self._selected_monitor_index)
            count = len(self._monitors)

        self._statusbar.showMessage(
            f"Found {count} device{'s' if count != 1 else ''}"
        )

    # ── Cleanup ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Clean up threads and resources on window close."""
        self._stop_capture()

        self._worker_thread.quit()
        self._worker_thread.wait(3000)

        if self._engine:
            self._engine.unload()

        event.accept()
