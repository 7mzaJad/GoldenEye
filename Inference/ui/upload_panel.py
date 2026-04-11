"""
Image upload panel — allows the user to select an image file,
displays the annotated prediction result.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QFileDialog, QSizePolicy,
)


class UploadPanel(QFrame):
    """Image upload and prediction result display."""

    # Emitted with the file path when user selects an image
    image_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header row
        header_row = QHBoxLayout()
        header = QLabel("📷  Image Upload")
        header.setObjectName("section-header")
        header_row.addWidget(header)
        header_row.addStretch()

        self._upload_btn = QPushButton("  Upload Image  ")
        self._upload_btn.setObjectName("primary")
        self._upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._upload_btn.clicked.connect(self._on_upload_clicked)
        header_row.addWidget(self._upload_btn)
        layout.addLayout(header_row)

        # Preview display
        self._display = QLabel()
        self._display.setObjectName("preview-placeholder")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._display.setMinimumSize(480, 300)
        self._display.setText("No image loaded\n\nDrag & drop or click Upload")
        layout.addWidget(self._display)

        # Info label
        self._info = QLabel("")
        self._info.setObjectName("subtitle")
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info)

    def _on_upload_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif);;All Files (*)",
        )
        if file_path:
            self.image_selected.emit(file_path)

    def show_result(self, annotated: np.ndarray, num_detections: int,
                    inference_ms: float) -> None:
        """Display the annotated prediction result."""
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)

        if num_detections > 0:
            self._info.setText(
                f"✓ {num_detections} human{'s' if num_detections != 1 else ''} detected "
                f"  •  {inference_ms:.1f}ms inference"
            )
            self._info.setStyleSheet("color: #10B981;")
        else:
            self._info.setText("No humans detected")
            self._info.setStyleSheet("color: #8B8FA3;")

    def show_error(self, message: str) -> None:
        """Show error state."""
        self._display.clear()
        self._display.setText(f"⚠  {message}")
        self._info.setText("")
        self._display.setStyleSheet("color: #EF4444; border-color: #5C2020;")

    def show_loading(self) -> None:
        """Show loading state during inference."""
        self._info.setText("Running inference…")
        self._info.setStyleSheet("color: #F5BD1F;")
