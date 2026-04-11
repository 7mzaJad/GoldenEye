"""
Camera preview panel — displays the live camera feed with
bounding box overlays rendered in the display coordinate space.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QFrame, QSizePolicy


class LiveViewPanel(QFrame):
    """Live preview widget with detection overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Section header
        header = QLabel("⦿  Live Detection Feed")
        header.setObjectName("section-header")
        layout.addWidget(header)

        # Preview display
        self._display = QLabel()
        self._display.setObjectName("preview-placeholder")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._display.setMinimumSize(480, 360)
        self._display.setText("Live Feed Off\n\nPress 'Start Capture' to begin")
        layout.addWidget(self._display)

    def update_frame(self, frame: np.ndarray) -> None:
        """Display a BGR frame (already annotated with bounding boxes)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        # .copy() prevents memory corruption if the numpy array is garbage collected
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(img)

        # Scale to fit the display label while keeping aspect ratio
        scaled = pixmap.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)

    def show_placeholder(self, text: str = "Live Feed Off\n\nPress 'Start Capture' to begin") -> None:
        """Reset display to placeholder text."""
        self._display.clear()
        self._display.setText(text)

    def show_error(self, message: str) -> None:
        """Show an error state on the preview."""
        self._display.clear()
        self._display.setText(f"⚠  {message}")
        self._display.setObjectName("preview-placeholder")
        self._display.setStyleSheet("color: #EF4444; border-color: #5C2020;")
