"""
Controls panel — model selector, start/stop buttons,
confidence/IoU sliders, and status indicators.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QSizePolicy,
    QRadioButton, QButtonGroup
)


class ControlsPanel(QFrame):
    """Application controls: model selection, thresholds, camera buttons."""

    # Signals emitted on user interaction
    start_capture = Signal()
    stop_capture = Signal()
    model_changed = Signal(str)           # model key
    device_changed = Signal(int)          # shared index for camera/monitor
    source_changed = Signal(str)          # 'camera' or 'screen'
    inference_mode_changed = Signal(str)  # 'standard' or 'sahi'
    conf_threshold_changed = Signal(float)
    iou_threshold_changed = Signal(float)

    def __init__(self, model_keys: list[str], defaults: dict,
                 devices: list | None = None,
                 inference_defaults: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._model_keys = model_keys
        self._defaults = defaults
        self._devices = devices or []
        self._inference_defaults = inference_defaults or {}
        self._current_source = "camera"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel("GoldenEye")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Search and Rescue Human Detection")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        self._add_separator(layout)

        # ── Input Source ───────────────────────────────────────────────
        layout.addWidget(self._make_label("Input Source"))
        
        self._source_group = QButtonGroup(self)
        self._radio_camera = QRadioButton("Camera")
        self._radio_camera.setChecked(True)
        self._radio_screen = QRadioButton("Screen")
        
        self._source_group.addButton(self._radio_camera, 0)
        self._source_group.addButton(self._radio_screen, 1)
        self._source_group.idToggled.connect(self._on_source_toggled)

        source_row = QHBoxLayout()
        source_row.addWidget(self._radio_camera)
        source_row.addWidget(self._radio_screen)
        layout.addLayout(source_row)

        # ── Model Selector ─────────────────────────────────────────────
        layout.addWidget(self._make_label("Model"))
        self._model_combo = QComboBox()
        for key in self._model_keys:
            self._model_combo.addItem(key.upper(), key)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self._model_combo)

        # ── Inference Mode ────────────────────────────────────────────
        layout.addWidget(self._make_label("Inference Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Standard", "standard")
        self._mode_combo.addItem("SAHI", "sahi")
        self._mode_combo.currentIndexChanged.connect(self._on_inference_mode_changed)
        layout.addWidget(self._mode_combo)

        default_mode = str(self._inference_defaults.get("mode", "standard")).lower()
        self.set_inference_mode(default_mode)

        self._add_separator(layout)

        # ── Device Controls ────────────────────────────────────────────
        self._device_label = self._make_label("Camera Device")
        layout.addWidget(self._device_label)

        # Device selector
        self._device_combo = QComboBox()
        self._populate_device_combo()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self._device_combo)

        # Refresh button (re-scan devices)
        self._refresh_btn = QPushButton("⟳  Rescan Devices")
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.setToolTip("Re-scan for connected cameras")
        layout.addWidget(self._refresh_btn)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._start_btn = QPushButton("▶  Start Capture")
        self._start_btn.setObjectName("primary")
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self.start_capture.emit)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop Capture")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_capture.emit)
        btn_row.addWidget(self._stop_btn)

        layout.addLayout(btn_row)

        self._add_separator(layout)

        # ── Confidence Threshold ───────────────────────────────────────
        conf_default = self._defaults.get("confidence_threshold", 0.25)
        self._conf_label = self._make_label(f"Confidence: {conf_default:.0%}")
        layout.addWidget(self._conf_label)

        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(5, 95)
        self._conf_slider.setValue(int(conf_default * 100))
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        layout.addWidget(self._conf_slider)

        # ── IoU Threshold ──────────────────────────────────────────────
        iou_default = self._defaults.get("iou_threshold", 0.45)
        self._iou_label = self._make_label(f"IoU (NMS): {iou_default:.0%}")
        layout.addWidget(self._iou_label)

        self._iou_slider = QSlider(Qt.Orientation.Horizontal)
        self._iou_slider.setRange(10, 95)
        self._iou_slider.setValue(int(iou_default * 100))
        self._iou_slider.valueChanged.connect(self._on_iou_changed)
        layout.addWidget(self._iou_slider)

        self._add_separator(layout)

        # ── Status Indicators ──────────────────────────────────────────
        layout.addWidget(self._make_label("Status"))

        self._status_model = self._make_status_row("Model", "Not loaded")
        layout.addLayout(self._status_model["layout"])

        self._status_camera = self._make_status_row("Camera", "Inactive")
        layout.addLayout(self._status_camera["layout"])

        self._status_inference = self._make_status_row("Inference", "Idle")
        layout.addLayout(self._status_inference["layout"])

        layout.addStretch()

    # ── Public API ─────────────────────────────────────────────────────

    def set_capture_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._device_combo.setEnabled(not running)
        self._refresh_btn.setEnabled(not running)
        self._radio_camera.setEnabled(not running)
        self._radio_screen.setEnabled(not running)
        
        self._update_status(
            self._status_camera,
            "Active" if running else "Inactive",
            "active" if running else "inactive",
        )

    def set_model_loaded(self, loaded: bool, name: str = "") -> None:
        text = f"Loaded ({name})" if loaded else "Not loaded"
        state = "active" if loaded else "inactive"
        self._update_status(self._status_model, text, state)

    def set_inference_state(self, state: str) -> None:
        """state: 'running', 'idle', 'error'"""
        display = {"running": "Running ⟳", "idle": "Idle", "error": "Error"}
        style = {"running": "active", "idle": "inactive", "error": "error"}
        self._update_status(
            self._status_inference,
            display.get(state, state),
            style.get(state, "inactive"),
        )

    def set_model_loading(self) -> None:
        self._update_status(self._status_model, "Loading…", "inactive")

    @property
    def selected_device_index(self) -> int:
        """Return the device/monitor index of the currently selected source."""
        data = self._device_combo.currentData()
        return data if data is not None else -1

    @property
    def refresh_button(self) -> QPushButton:
        """Expose the rescan button so MainWindow can connect to it."""
        return self._refresh_btn

    def set_devices(self, devices: list) -> None:
        """Update the device dropdown with a new device list."""
        self._devices = devices
        self._populate_device_combo()

    def select_device_index(self, index: int) -> None:
        """Programmatically select a device by index."""
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == index:
                self._device_combo.setCurrentIndex(i)
                return

    def set_inference_mode(self, mode: str) -> None:
        """Programmatically select inference mode in the dropdown."""
        normalized = str(mode).lower().strip()
        for i in range(self._mode_combo.count()):
            if self._mode_combo.itemData(i) == normalized:
                self._mode_combo.setCurrentIndex(i)
                return

    @property
    def inference_mode(self) -> str:
        data = self._mode_combo.currentData()
        return str(data) if data is not None else "standard"

    @property
    def confidence_threshold(self) -> float:
        return self._conf_slider.value() / 100.0

    @property
    def iou_threshold(self) -> float:
        return self._iou_slider.value() / 100.0

    # ── Internal ───────────────────────────────────────────────────────

    def _on_model_changed(self, index: int) -> None:
        key = self._model_combo.itemData(index)
        self.model_changed.emit(key)

    def _on_device_changed(self, index: int) -> None:
        data = self._device_combo.itemData(index)
        if data is not None:
            self.device_changed.emit(data)

    def _on_source_toggled(self, btn_id: int, checked: bool) -> None:
        if checked:
            new_source = "camera" if btn_id == 0 else "screen"
            if new_source != self._current_source:
                self._current_source = new_source
                self._device_label.setText("Camera Device" if new_source == "camera" else "Display Monitor")
                self._status_camera["value"].setText("Camera:" if new_source == "camera" else "Screen capture:")
                self.source_changed.emit(new_source)

    def _on_conf_changed(self, value: int) -> None:
        fval = value / 100.0
        self._conf_label.setText(f"Confidence: {fval:.0%}")
        self.conf_threshold_changed.emit(fval)

    def _on_inference_mode_changed(self, index: int) -> None:
        mode = self._mode_combo.itemData(index)
        if mode is not None:
            self.inference_mode_changed.emit(str(mode))

    def _on_iou_changed(self, value: int) -> None:
        fval = value / 100.0
        self._iou_label.setText(f"IoU (NMS): {fval:.0%}")
        self.iou_threshold_changed.emit(fval)

    def _populate_device_combo(self) -> None:
        """Fill the generic device dropdown from the current list."""
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        if self._devices:
            for dev in self._devices:
                self._device_combo.addItem(dev.display_name, dev.index)
        else:
            msg = "No cameras found" if self._current_source == "camera" else "No screens found"
            self._device_combo.addItem(msg, -1)
        self._device_combo.blockSignals(False)

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section-header")
        return lbl

    @staticmethod
    def _make_status_row(name: str, initial: str) -> dict:
        row = QHBoxLayout()
        row.setSpacing(8)

        dot = QLabel("●")
        dot.setObjectName("status-dot-inactive")
        dot.setFixedWidth(18)
        row.addWidget(dot)

        label = QLabel(f"{name}:")
        label.setStyleSheet("color: #8B8FA3; font-size: 12px;")
        label.setFixedWidth(72)
        row.addWidget(label)

        value = QLabel(initial)
        value.setStyleSheet("color: #F0F1F5; font-size: 12px; font-weight: 500;")
        row.addWidget(value)
        row.addStretch()

        return {"layout": row, "dot": dot, "value": value}

    @staticmethod
    def _update_status(status: dict, text: str, state: str) -> None:
        status["value"].setText(text)
        status["dot"].setObjectName(f"status-dot-{state}")
        # Force stylesheet re-evaluation
        status["dot"].style().unpolish(status["dot"])
        status["dot"].style().polish(status["dot"])

    @staticmethod
    def _add_separator(layout: QVBoxLayout) -> None:
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
