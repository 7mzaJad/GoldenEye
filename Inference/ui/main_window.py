"""
Main application window — orchestrates all panels, camera thread,
and inference engine into a cohesive real-time SAR human detection interface.
"""

import logging
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStatusBar, QMessageBox,
)

from core.engine import InferenceEngine
from handlers.camera import CameraThread, scan_cameras, get_preferred_camera
from handlers.image import ImageHandler
from utils.config_loader import load_config, resolve_model_path
from utils.drawing import draw_detections, Detection
from utils.constants import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT

from ui.camera_panel import CameraPanel
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

    result_ready = Signal(np.ndarray, list, float)  # annotated_frame, detections, ms
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._engine: InferenceEngine | None = None
        self._conf_threshold = 0.25
        self._iou_threshold = 0.45
        self._busy = False

    def set_engine(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def set_thresholds(self, conf: float, iou: float) -> None:
        self._conf_threshold = conf
        self._iou_threshold = iou

    @property
    def is_busy(self) -> bool:
        return self._busy

    @Slot(np.ndarray)
    def process_frame(self, frame: np.ndarray) -> None:
        """Run inference on a single frame. Skips if already busy."""
        if self._busy or self._engine is None or not self._engine.is_loaded:
            return

        self._busy = True
        try:
            detections, inference_ms = self._engine.predict(
                frame, self._conf_threshold, self._iou_threshold
            )
            annotated = frame.copy()
            draw_detections(annotated, detections)
            self.result_ready.emit(annotated, detections, inference_ms)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._busy = False


# ── Main Window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """GoldenEye main application window."""

    # Internal signal to send frames to the inference worker thread
    _send_frame = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GoldenEye — Real-Time Search and Rescue Human Detection")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # State
        self._config = load_config()
        self._engine: InferenceEngine | None = None
        self._camera_thread: CameraThread | None = None
        self._image_handler: ImageHandler | None = None
        self._fps_times: deque[float] = deque(maxlen=30)
        self._cameras: list = []           # detected CameraDevice list
        self._selected_cam_index: int = 0  # currently selected device index

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        # Auto-scan connected cameras
        self._scan_cameras()

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
            model_keys, self._config["defaults"], cameras=self._cameras
        )
        self._controls.setFixedWidth(280)
        self._controls.start_camera.connect(self._start_camera)
        self._controls.stop_camera.connect(self._stop_camera)
        self._controls.model_changed.connect(self._load_model)
        self._controls.camera_changed.connect(self._on_camera_device_changed)
        self._controls.conf_threshold_changed.connect(self._on_conf_changed)
        self._controls.iou_threshold_changed.connect(self._on_iou_changed)
        self._controls.refresh_button.clicked.connect(self._rescan_cameras)

        # Auto-select the preferred camera in the dropdown
        self._controls.select_camera_index(self._selected_cam_index)

        root_layout.addWidget(self._controls)

        # ── Center: Camera + Upload ────────────────────────────────────
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)

        self._camera_panel = CameraPanel()
        center_layout.addWidget(self._camera_panel, stretch=3)

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
        self._worker = InferenceWorker()
        self._worker.moveToThread(self._worker_thread)

        # Connect signals
        self._send_frame.connect(self._worker.process_frame)
        self._worker.result_ready.connect(self._on_inference_result)
        self._worker.error.connect(self._on_inference_error)

        self._worker_thread.start()

    # ── Model Loading ──────────────────────────────────────────────────

    def _load_model(self, model_key: str) -> None:
        """Load an ONNX model by its config key."""
        self._controls.set_model_loading()
        self._statusbar.showMessage(f"Loading model: {model_key}…")

        # Stop camera while switching models
        if self._camera_thread and self._camera_thread.is_running:
            self._stop_camera()

        # Unload previous engine
        if self._engine:
            self._engine.unload()

        try:
            model_cfg = self._config["models"][model_key]
            model_path = resolve_model_path(self._config, model_key)

            self._engine = InferenceEngine(
                model_path=model_path,
                model_type=model_cfg["type"],
                class_names=self._config["class_names"],
                input_size=self._config["defaults"].get("input_size", 640),
            )
            self._engine.load()

            # Update worker
            self._worker.set_engine(self._engine)
            self._image_handler = ImageHandler(self._engine)

            self._controls.set_model_loaded(True, model_key.upper())
            self._statusbar.showMessage(
                f"Model loaded: {model_key.upper()} — Ready for detection"
            )

        except (FileNotFoundError, RuntimeError) as e:
            self._controls.set_model_loaded(False)
            self._statusbar.showMessage("Model loading failed")
            QMessageBox.critical(
                self, "Model Error",
                f"Failed to load model '{model_key}':\n\n{e}"
            )

    # ── Camera Controls ────────────────────────────────────────────────

    def _start_camera(self) -> None:
        if not self._engine or not self._engine.is_loaded:
            QMessageBox.warning(
                self, "No Model",
                "Please load a model before starting the camera."
            )
            return

        cam_index = self._controls.selected_camera_index
        if cam_index < 0:
            QMessageBox.warning(
                self, "No Camera",
                "No camera device selected.\nClick 'Rescan Devices' to detect cameras."
            )
            return

        self._camera_thread = CameraThread(cam_index)
        self._camera_thread.frame_ready.connect(self._on_camera_frame)
        self._camera_thread.error.connect(self._on_camera_error)
        self._camera_thread.start()

        self._fps_times.clear()
        self._controls.set_camera_running(True)
        self._controls.set_inference_state("running")

        # Find the device name for the status bar
        dev_name = f"Camera {cam_index}"
        for dev in self._cameras:
            if dev.index == cam_index:
                dev_name = dev.name
                break
        self._statusbar.showMessage(
            f"Camera active — {dev_name} — Running real-time detection"
        )

    def _stop_camera(self) -> None:
        if self._camera_thread:
            self._camera_thread.frame_ready.disconnect(self._on_camera_frame)
            self._camera_thread.stop()
            self._camera_thread = None

        self._controls.set_camera_running(False)
        self._controls.set_inference_state("idle")
        self._camera_panel.show_placeholder()
        self._results.clear()
        self._statusbar.showMessage("Camera stopped")

    # ── Signal Handlers ────────────────────────────────────────────────

    @Slot(np.ndarray)
    def _on_camera_frame(self, frame: np.ndarray) -> None:
        """Received a new camera frame — send to inference worker."""
        self._send_frame.emit(frame)

    @Slot(np.ndarray, list, float)
    def _on_inference_result(
        self, annotated: np.ndarray, detections: list, inference_ms: float
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

        self._camera_panel.update_frame(annotated)
        self._results.update_results(detections, inference_ms, fps)

    @Slot(str)
    def _on_inference_error(self, message: str) -> None:
        logger.error(f"Inference error: {message}")
        self._controls.set_inference_state("error")
        self._statusbar.showMessage(f"Inference error: {message}")

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        self._controls.set_camera_running(False)
        self._controls.set_inference_state("idle")
        self._camera_panel.show_error(message)
        QMessageBox.warning(self, "Camera Error", message)

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
            _, annotated, detections, inference_ms = self._image_handler.predict(
                file_path, conf, iou
            )
            self._upload_panel.show_result(annotated, len(detections), inference_ms)

            # Also update results panel if camera is not running
            if not (self._camera_thread and self._camera_thread.is_running):
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

    def _on_camera_device_changed(self, index: int) -> None:
        """User picked a different camera from the dropdown."""
        self._selected_cam_index = index
        logger.info(f"Camera selection changed to index {index}")

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

    def _rescan_cameras(self) -> None:
        """Re-scan devices and refresh the dropdown (called from UI button)."""
        self._statusbar.showMessage("Scanning for cameras…")
        self._scan_cameras()
        self._controls.set_cameras(self._cameras)
        self._controls.select_camera_index(self._selected_cam_index)

        count = len(self._cameras)
        self._statusbar.showMessage(
            f"Found {count} camera{'s' if count != 1 else ''}"
        )

    # ── Cleanup ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Clean up threads and resources on window close."""
        self._stop_camera()

        self._worker_thread.quit()
        self._worker_thread.wait(3000)

        if self._engine:
            self._engine.unload()

        event.accept()
