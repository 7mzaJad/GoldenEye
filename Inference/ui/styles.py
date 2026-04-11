"""
QSS stylesheet — dark theme with gold accents.
"""

MAIN_STYLESHEET = """
/* ── Global ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #0F1117;
    color: #F0F1F5;
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* ── Cards ──────────────────────────────────────────────────────────── */
QFrame#card {
    background-color: #1A1D27;
    border: 1px solid #2E3140;
    border-radius: 12px;
    padding: 16px;
}
QFrame#card:hover {
    border-color: #3A3D50;
}

/* ── Labels ─────────────────────────────────────────────────────────── */
QLabel {
    color: #F0F1F5;
    background: transparent;
}
QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #F5BD1F;
    padding: 0;
}
QLabel#subtitle {
    font-size: 12px;
    font-weight: 400;
    color: #8B8FA3;
    padding: 0;
}
QLabel#section-header {
    font-size: 14px;
    font-weight: 600;
    color: #F0F1F5;
    padding: 4px 0;
}
QLabel#stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #F5BD1F;
}
QLabel#stat-label {
    font-size: 11px;
    color: #8B8FA3;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLabel#status-dot-active {
    color: #10B981;
    font-size: 14px;
}
QLabel#status-dot-inactive {
    color: #5C5F72;
    font-size: 14px;
}
QLabel#status-dot-error {
    color: #EF4444;
    font-size: 14px;
}
QLabel#preview-placeholder {
    color: #5C5F72;
    font-size: 15px;
    font-weight: 500;
    background-color: #14161E;
    border: 2px dashed #2E3140;
    border-radius: 10px;
    padding: 40px;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #22252F;
    color: #F0F1F5;
    border: 1px solid #2E3140;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #2A2D38;
    border-color: #F5BD1F;
    color: #F5BD1F;
}
QPushButton:pressed {
    background-color: #1A1D27;
}
QPushButton:disabled {
    background-color: #14161E;
    color: #5C5F72;
    border-color: #1E2030;
}

QPushButton#primary {
    background-color: #F5BD1F;
    color: #0F1117;
    border: none;
    font-weight: 700;
}
QPushButton#primary:hover {
    background-color: #FFD54F;
    color: #0F1117;
}
QPushButton#primary:pressed {
    background-color: #D4A017;
}
QPushButton#primary:disabled {
    background-color: #3A3520;
    color: #7A7060;
}

QPushButton#danger {
    background-color: #2D1518;
    color: #EF4444;
    border: 1px solid #5C2020;
}
QPushButton#danger:hover {
    background-color: #3D1D20;
    border-color: #EF4444;
}

/* ── Sliders ────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #22252F;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #F5BD1F;
    border: 2px solid #D4A017;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #FFD54F;
    border-color: #F5BD1F;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #F5BD1F, stop:1 #D4A017);
    border-radius: 3px;
}

/* ── ComboBox ───────────────────────────────────────────────────────── */
QComboBox {
    background-color: #22252F;
    color: #F0F1F5;
    border: 1px solid #2E3140;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #F5BD1F;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #8B8FA3;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1A1D27;
    color: #F0F1F5;
    border: 1px solid #2E3140;
    selection-background-color: #F5BD1F;
    selection-color: #0F1117;
    border-radius: 8px;
    padding: 4px;
}

/* ── Scroll Area ────────────────────────────────────────────────────── */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #14161E;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2E3140;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #3A3D50;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── Table ──────────────────────────────────────────────────────────── */
QTableWidget {
    background-color: #14161E;
    color: #F0F1F5;
    border: 1px solid #2E3140;
    border-radius: 8px;
    gridline-color: #1E2030;
    font-size: 12px;
}
QTableWidget::item {
    padding: 6px 10px;
    border-bottom: 1px solid #1E2030;
}
QTableWidget::item:selected {
    background-color: #2A2520;
    color: #F5BD1F;
}
QHeaderView::section {
    background-color: #1A1D27;
    color: #8B8FA3;
    border: none;
    border-bottom: 1px solid #2E3140;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

/* ── Status Bar ─────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #14161E;
    color: #8B8FA3;
    border-top: 1px solid #1E2030;
    font-size: 11px;
    padding: 4px 12px;
}

/* ── Separator ──────────────────────────────────────────────────────── */
QFrame#separator {
    background-color: #2E3140;
    max-height: 1px;
    margin: 8px 0;
}

/* ── Tooltips ───────────────────────────────────────────────────────── */
QToolTip {
    background-color: #1A1D27;
    color: #F0F1F5;
    border: 1px solid #2E3140;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
"""
