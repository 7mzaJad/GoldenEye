"""
Application-wide constants: colors, fonts, metadata.
"""

APP_NAME = "GoldenEye"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Real-Time Search and Rescue Human Detection System"

WINDOW_WIDTH = 1320
WINDOW_HEIGHT = 820
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 700

# ── Color Palette ──────────────────────────────────────────────────────────
# Detection bounding box colors — one per class, wraps around
DETECTION_COLORS = [
    (245, 189, 31),    # Gold / Amber
    (59, 130, 246),    # Blue
    (16, 185, 129),    # Emerald
    (239, 68, 68),     # Red
    (168, 85, 247),    # Purple
    (236, 72, 153),    # Pink
    (251, 146, 60),    # Orange
    (6, 182, 212),     # Cyan
]

# UI Accent colors (hex for QSS)
ACCENT_GOLD = "#F5BD1F"
ACCENT_GOLD_DARK = "#D4A017"
ACCENT_GOLD_LIGHT = "#FFD54F"
BG_PRIMARY = "#0F1117"
BG_SECONDARY = "#1A1D27"
BG_CARD = "#22252F"
BG_CARD_HOVER = "#2A2D38"
BORDER_SUBTLE = "#2E3140"
TEXT_PRIMARY = "#F0F1F5"
TEXT_SECONDARY = "#8B8FA3"
TEXT_MUTED = "#5C5F72"
SUCCESS_GREEN = "#10B981"
ERROR_RED = "#EF4444"
WARNING_AMBER = "#F59E0B"

# ── Drawing Defaults ───────────────────────────────────────────────────────
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1
LABEL_PADDING = 4
LABEL_BG_ALPHA = 0.7
