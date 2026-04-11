"""
GoldenEye — Real-Time Search and Rescue Human Detection
Application Entry Point

Usage:
    python main.py
"""

import sys
import os
import logging

# Ensure the Inference directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ui.main_window import MainWindow
from utils.constants import APP_NAME, APP_VERSION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info(f"{APP_NAME} v{APP_VERSION} starting…")

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Default font
    font = QFont("SF Pro Display", 13)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Launch main window
    window = MainWindow()
    window.show()

    logger.info("Application window opened")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
