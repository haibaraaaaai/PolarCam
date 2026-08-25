"""
TEST ONLY — AVI mock camera backend.
Swap this in to test the full GUI pipeline without a real camera.
Easy to remove: delete this file and remove the "Test (AVI)" button from main_window.py.

Emits the same signals as IDSCamera so the entire app is unaware it is fake.
Frames are delivered as uint16 (values 0-4095).
Control calls (set_roi, set_timing, set_gains…) are silently ignored, which means:
  - SpotCycler's maximize_fps call is harmless.
  - ROI requests do nothing; full frames continue to arrive.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt

from polarcam.backend.base import ICamera

# -- optional cv2 import for AVI decode -----------------------------------
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    _CV2_OK = False


def _load_avi_frames(path: str) -> list[np.ndarray]:
    """
    Load every frame of an AVI into a list of uint16 (H, W) arrays.
    Values are scaled to the 12-bit range 0-4095.
    Returns an empty list on failure.
    """
    if not _CV2_OK:
        raise RuntimeError(
            "opencv-python is not installed. Run:\n"
            "  pip install opencv-python-headless"
        )
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # (H, W) uint8
        frames.append((gray.astype(np.uint16) << 4))   # scale to 12-bit
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Emission worker (lives on its own QThread)
# ---------------------------------------------------------------------------

class _LoopWorker(QObject):
    frame = Signal(object)

    def __init__(self, frames: list[np.ndarray], fps: float = 20.0) -> None:
        super().__init__()
        self._frames = frames
        self._period = 1.0 / max(1.0, float(fps))
        self._running = False

    def set_fps(self, fps: float) -> None:
        self._period = 1.0 / max(1.0, float(fps))

    @Slot()
    def run(self) -> None:
        self._running = True
        n = len(self._frames)
        i = 0
        while self._running:
            t0 = time.perf_counter()
            self.frame.emit(self._frames[i % n])
            i += 1
            elapsed = time.perf_counter() - t0
            sleep = self._period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    @Slot()
    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Public mock camera
# ---------------------------------------------------------------------------

class AviMockCamera(ICamera):
    """
    Fake camera that loops frames from an AVI file.

    Usage::

        mock = AviMockCamera()
        mock.open("path/to/file.avi")
        mock.start()
        # connect mock.frame to anything that expects cam.frame
        ...
        mock.stop()
        mock.close()
    """

    def __init__(self) -> None:
        super().__init__()
        self._frames: list[np.ndarray] = []
        self._worker: Optional[_LoopWorker] = None
        self._thread: Optional[QThread] = None
        self._path = ""

    # ------------------------------------------------------------------
    # ICamera interface
    # ------------------------------------------------------------------

    def open(self, path: str = "") -> None:  # type: ignore[override]
        self._path = str(path)
        self._frames = _load_avi_frames(self._path)
        if not self._frames:
            self.error.emit(f"AVI mock: no frames loaded from {self._path!r}")
            return
        H, W = self._frames[0].shape
        self.opened.emit(f"[MOCK] {self._path}  ({len(self._frames)} frames, {W}×{H})")
        # Emit fake ROI so the rest of the UI populates correctly
        self.roi.emit({"Width": W, "Height": H, "OffsetX": 0, "OffsetY": 0})
        self.timing.emit({"fps": 20.0, "exposure_us": 50_000})

    def start(self) -> None:
        if not self._frames:
            self.error.emit("AVI mock: call open() first.")
            return
        if self._thread is not None:
            return  # already running
        self._worker = _LoopWorker(self._frames, fps=20.0)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame.connect(self.frame, Qt.QueuedConnection)
        self._thread.start()
        self.started.emit()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = None
        self._thread = None
        self.stopped.emit()

    def close(self) -> None:
        self.stop()
        self._frames = []
        self.closed.emit()

    # ------------------------------------------------------------------
    # Control calls — silently ignored (no-ops)
    # ------------------------------------------------------------------

    def set_roi(self, w, h, x, y) -> None:  # type: ignore[override]
        pass

    def full_sensor(self) -> None:
        pass

    def set_timing(self, fps=None, exposure_ms=None) -> None:  # type: ignore[override]
        # Silently ignore — looping continues at original rate.
        pass

    def set_zoom_roi(self, roi_xywh=None) -> None:  # type: ignore[override]
        pass

    def set_gains(self, analog, digital) -> None:  # type: ignore[override]
        pass

    def refresh_gains(self) -> None:
        pass

    def refresh_timing(self) -> None:
        pass

    def auto_desaturate(self, *args, **kwargs) -> None:  # type: ignore[override]
        pass
