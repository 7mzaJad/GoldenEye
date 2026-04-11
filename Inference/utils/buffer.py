import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker

class FrameBuffer:
    """A thread-safe single-slot container for the latest captured frame.
    
    Prevents unbounded queue growth by purely overwriting the previous frame.
    """
    def __init__(self):
        self._mutex = QMutex()
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray) -> None:
        """Write the latest frame, destroying the previous one."""
        with QMutexLocker(self._mutex):
            self._frame = frame

    def get(self) -> np.ndarray | None:
        """Retrieve the latest frame and clear the slot."""
        with QMutexLocker(self._mutex):
            frame = self._frame
            self._frame = None
            return frame

    def clear(self) -> None:
        """Clear the buffer."""
        with QMutexLocker(self._mutex):
            self._frame = None
