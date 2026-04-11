"""
Results panel — detection table, stats counters, and timing display.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy,
)

from utils.drawing import Detection


class ResultsPanel(QFrame):
    """Detection results table and summary statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── Header ─────────────────────────────────────────────────────
        header = QLabel("📊  Detection Results")
        header.setObjectName("section-header")
        layout.addWidget(header)

        # ── Row 1: Detection Stats ─────────────────────────────────────
        det_row = QHBoxLayout()
        det_row.setSpacing(12)

        self._stat_detections = self._make_stat_card("0", "HUMANS DETECTED")
        det_row.addWidget(self._stat_detections["frame"])

        self._stat_max_conf = self._make_stat_card("—", "MAX CONFIDENCE")
        det_row.addWidget(self._stat_max_conf["frame"])

        self._stat_avg_conf = self._make_stat_card("—", "AVG CONFIDENCE")
        det_row.addWidget(self._stat_avg_conf["frame"])

        layout.addLayout(det_row)

        # ── Row 2: Performance Stats ───────────────────────────────────
        perf_row = QHBoxLayout()
        perf_row.setSpacing(12)

        self._stat_latency = self._make_stat_card("—", "INFERENCE (ms)")
        perf_row.addWidget(self._stat_latency["frame"])

        self._stat_fps = self._make_stat_card("—", "FPS")
        perf_row.addWidget(self._stat_fps["frame"])

        layout.addLayout(perf_row)

        layout.addStretch()

        # ── Empty state note ───────────────────────────────────────────
        self._empty_label = QLabel("No humans detected yet")
        self._empty_label.setObjectName("subtitle")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

    # ── Public API ─────────────────────────────────────────────────────

    def update_results(
        self,
        detections: list[Detection],
        inference_ms: float,
        fps: float | None = None,
    ) -> None:
        """Update all result displays with new detection data."""
        num = len(detections)

        # Performance
        self._stat_latency["value"].setText(f"{inference_ms:.1f}")
        if fps is not None:
            self._stat_fps["value"].setText(f"{fps:.1f}")

        # Empty state toggle
        self._empty_label.setVisible(num == 0)

        if num == 0:
            self._stat_detections["value"].setText("0")
            self._stat_max_conf["value"].setText("—")
            self._stat_avg_conf["value"].setText("—")
            return

        # SAR Metrics
        max_conf = max(d.confidence for d in detections)
        avg_conf = sum(d.confidence for d in detections) / num

        self._stat_detections["value"].setText(str(num))
        self._stat_max_conf["value"].setText(f"{max_conf:.1%}")
        self._stat_avg_conf["value"].setText(f"{avg_conf:.1%}")

    def clear(self) -> None:
        """Reset to empty state."""
        self._stat_detections["value"].setText("0")
        self._stat_max_conf["value"].setText("—")
        self._stat_avg_conf["value"].setText("—")
        self._stat_latency["value"].setText("—")
        self._stat_fps["value"].setText("—")
        self._empty_label.setVisible(True)

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _make_stat_card(value: str, label: str) -> dict:
        frame = QFrame()
        frame.setObjectName("card")
        frame.setStyleSheet(
            "QFrame#card { background-color: #14161E; padding: 12px; }"
        )
        vlayout = QVBoxLayout(frame)
        vlayout.setContentsMargins(12, 10, 12, 10)
        vlayout.setSpacing(2)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        val = QLabel(value)
        val.setObjectName("stat-value")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(val)

        lbl = QLabel(label)
        lbl.setObjectName("stat-label")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(lbl)

        return {"frame": frame, "value": val, "label": lbl}
