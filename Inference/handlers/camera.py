"""
Camera capture thread and device discovery.

Runs in a separate QThread to avoid blocking the GUI.
Writes frames to a shared thread-safe FrameBuffer and emits a lightweight
signal for the inference worker to consume, completely preventing
endless queueing and "time-travel" latency.

Also provides scan_cameras() for auto-detecting available devices
and get_preferred_camera() for iPhone / Continuity Camera preference.
"""

import logging
import platform
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from utils.buffer import FrameBuffer

logger = logging.getLogger(__name__)

# Keywords that identify an iPhone / Continuity Camera device
_IPHONE_KEYWORDS = ("iphone", "continuity", "ios", "apple")


# ── Camera Device Discovery ──────────────────────────────────────────────

@dataclass
class CameraDevice:
    """Represents a detected camera device."""
    index: int
    name: str
    is_iphone: bool = False

    @property
    def display_name(self) -> str:
        tag = " 📱" if self.is_iphone else ""
        return f"[{self.index}] {self.name}{tag}"


def _get_macos_camera_names() -> dict[int, str]:
    """Use system_profiler on macOS to map indices to device names.

    Returns a dict mapping camera index → device name.  Falls back
    to empty dict if the command fails.
    """
    if platform.system() != "Darwin":
        return {}

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=5,
        )
        # Parse the plain-text output for camera model names
        names: list[str] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            # Top-level camera entries are lines that end with ':'
            # and are not indented labels like "Model ID:" etc.
            if stripped.endswith(":") and ":" not in stripped[:-1]:
                names.append(stripped[:-1])
        return {i: name for i, name in enumerate(names)}
    except Exception as e:
        logger.debug(f"system_profiler failed: {e}")
        return {}


def scan_cameras(max_index: int = 5) -> list[CameraDevice]:
    """Probe camera indices 0–max_index and return available devices.

    On macOS, attempts to resolve human-readable names via
    system_profiler.  On other platforms, names are generic.

    Args:
        max_index: Highest device index to probe (inclusive).

    Returns:
        List of CameraDevice objects for cameras that opened successfully.
    """
    mac_names = _get_macos_camera_names()
    devices: list[CameraDevice] = []

    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Try to read a frame to confirm the device is real
            ret, _ = cap.read()
            cap.release()
            if not ret:
                continue

            name = mac_names.get(idx, f"Camera {idx}")
            is_iphone = any(kw in name.lower() for kw in _IPHONE_KEYWORDS)
            devices.append(CameraDevice(index=idx, name=name, is_iphone=is_iphone))
            logger.info(f"Found camera: [{idx}] {name}"
                        f"{' (iPhone)' if is_iphone else ''}")
        else:
            cap.release()

    if not devices:
        logger.warning("No cameras detected")

    return devices


def get_preferred_camera(
    devices: list[CameraDevice],
    fallback_index: int = 0,
) -> int:
    """Return the best camera index — prefers iPhone, falls back to default.

    Args:
        devices:        List from scan_cameras().
        fallback_index: Index to use if no iPhone is found and the list
                        doesn't contain a device at index 0.

    Returns:
        Camera device index to use.
    """
    # Prefer iPhone / Continuity Camera
    for dev in devices:
        if dev.is_iphone:
            logger.info(f"Auto-selected iPhone camera: [{dev.index}] {dev.name}")
            return dev.index

    # Fallback to the configured default or first available
    for dev in devices:
        if dev.index == fallback_index:
            return dev.index

    if devices:
        return devices[0].index

    return fallback_index


# ── Camera Capture Thread ─────────────────────────────────────────────────

class CameraThread(QThread):
    """Captures frames from a camera device in a background thread.

    Signals:
        frame_ready(): Emitted when a new frame is written to the buffer.
        error(str):    Emitted when camera access fails.
    """

    frame_ready = Signal()
    error = Signal(str)

    def __init__(self, camera_index: int, buffer: FrameBuffer, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._buffer = buffer
        self._running = False
        self._mutex = QMutex()
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def run(self) -> None:
        """Thread entry point — captures frames in a loop."""
        self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            self.error.emit(
                f"Cannot access camera (index {self._camera_index}).\n"
                "Please check that your camera is connected and not in use."
            )
            return

        # Optimize capture settings
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with QMutexLocker(self._mutex):
            self._running = True

        logger.info(f"Camera started (index {self._camera_index})")

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            ret, frame = self._cap.read()
            if not ret:
                continue

            self._buffer.put(frame)
            self.frame_ready.emit()

        self._cap.release()
        self._cap = None
        logger.info("Camera stopped")

    def stop(self) -> None:
        """Signal the capture loop to exit and wait for thread to finish."""
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(3000)
