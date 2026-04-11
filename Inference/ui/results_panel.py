"""
Results panel — detection table, stats counters, and timing display.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
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

        # ── Stats Row ──────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)

        self._stat_detections = self._make_stat_card("0", "HUMANS DETECTED")
        stats_row.addWidget(self._stat_detections["frame"])

        self._stat_latency = self._make_stat_card("—", "LATENCY (ms)")
        stats_row.addWidget(self._stat_latency["frame"])

        self._stat_fps = self._make_stat_card("—", "FPS")
        stats_row.addWidget(self._stat_fps["frame"])

        layout.addLayout(stats_row)

        # ── Detection Table ────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Class", "Confidence", "Count"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMinimumHeight(120)
        layout.addWidget(self._table)

        # ── Empty state ────────────────────────────────────────────────
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

        # Stats
        self._stat_detections["value"].setText(str(num))
        self._stat_latency["value"].setText(f"{inference_ms:.1f}")
        if fps is not None:
            self._stat_fps["value"].setText(f"{fps:.1f}")

        # Empty state
        self._empty_label.setVisible(num == 0)
        self._table.setVisible(num > 0)

        if num == 0:
            self._table.setRowCount(0)
            return

        # Aggregate by class
        class_counts: dict[str, dict] = {}
        for det in detections:
            if det.class_name not in class_counts:
                class_counts[det.class_name] = {
                    "count": 0,
                    "max_conf": 0.0,
                }
            class_counts[det.class_name]["count"] += 1
            class_counts[det.class_name]["max_conf"] = max(
                class_counts[det.class_name]["max_conf"], det.confidence
            )

        # Populate table
        self._table.setRowCount(len(class_counts))
        for row, (name, data) in enumerate(
            sorted(class_counts.items(), key=lambda x: -x[1]["count"])
        ):
            self._table.setItem(row, 0, self._make_item(name))
            self._table.setItem(
                row, 1, self._make_item(f"{data['max_conf']:.1%}")
            )
            self._table.setItem(
                row, 2, self._make_item(str(data["count"]))
            )

    def clear(self) -> None:
        """Reset to empty state."""
        self._table.setRowCount(0)
        self._stat_detections["value"].setText("0")
        self._stat_latency["value"].setText("—")
        self._stat_fps["value"].setText("—")
        self._empty_label.setVisible(True)
        self._table.setVisible(False)

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

    @staticmethod
    def _make_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        return item
