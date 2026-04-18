import logging
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from utils.buffer import FrameBuffer

logger = logging.getLogger(__name__)


@dataclass
class MonitorDevice:
    """Represents a detected monitor."""

    index: int
    name: str
    mss_monitor_dict: dict

    @property
    def display_name(self) -> str:
        return f"[{self.index}] {self.name}"


def scan_monitors() -> list[MonitorDevice]:
    """Return a list of available displays using mss."""
    devices = []
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            # mss.monitors[0] is the virtual bounding box of all monitors.
            for i, mon in enumerate(monitors):
                name = "All Displays" if i == 0 else f"Display {i}"
                devices.append(MonitorDevice(index=i, name=name, mss_monitor_dict=mon))
        logger.info(f"Detected {len(devices)} monitors.")
    except Exception as e:
        logger.error(f"Failed to scan monitors: {e}")
    return devices


class ScreenThread(QThread):
    """Captures frames from the screen in a background thread."""

    frame_ready = Signal()
    error = Signal(str)

    def __init__(
        self, monitor_dict: dict, buffer: FrameBuffer, fps: int = 30, parent=None
    ):
        super().__init__(parent)
        self._monitor = monitor_dict
        self._buffer = buffer
        self._running = False
        self._mutex = QMutex()
        self._target_interval = 1.0 / fps

    @property
    def is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def run(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = True

        logger.info(f"Screen capture started")

        try:
            with mss.mss() as sct:
                while True:
                    t0 = time.perf_counter()

                    with QMutexLocker(self._mutex):
                        if not self._running:
                            break

                    # Capture screen
                    sct_img = sct.grab(self._monitor)

                    # Convert to numpy array (BGRA)
                    img = np.array(sct_img)

                    # Convert BGRA to BGR to match OpenCV camera format
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    self._buffer.put(frame)
                    self.frame_ready.emit()

                    # Throttle to max target FPS
                    elapsed = time.perf_counter() - t0
                    sleep_time = self._target_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            self.error.emit(str(e))
        finally:
            logger.info("Screen capture stopped")

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(3000)
