from __future__ import annotations
"""
Thin application-facing controller.

- Owns the camera facade (IDSCamera) but stays UI-agnostic.
- Provides small convenience helpers (auto-desaturate).
- Avoids terminal spam: uses `logging` (configured in `polarcam.__init__`).
"""

import logging
from PySide6.QtCore import QObject

from polarcam.backend.ids_backend import IDSCamera

log = logging.getLogger(__name__)


class Controller(QObject):
    """High-level control surface used by the Qt widgets."""

    def __init__(self, cam: IDSCamera | None = None) -> None:
        super().__init__()
        self.cam = cam or IDSCamera()
        log.debug("Controller initialized with %s", type(self.cam).__name__)

    # ---------- lifecycle -------------------------------------------------
    def open(self) -> None:
        """Open the underlying camera device."""
        log.info("open()")
        self.cam.open()

    def start(self) -> None:
        """Start acquisition (idempotent on the facade)."""
        log.info("start()")
        self.cam.start()

    def stop(self) -> None:
        """Stop acquisition (safe to call when already stopped)."""
        log.info("stop()")
        self.cam.stop()

    def close(self) -> None:
        """Close the camera and tear down resources."""
        log.info("close()")
        self.cam.close()

    # ---------- controls --------------------------------------------------
    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        """Apply ROI (Width, Height, OffsetX, OffsetY) in sensor pixels."""
        log.debug("set_roi(w=%s, h=%s, x=%s, y=%s)", w, h, x, y)
        self.cam.set_roi(w, h, x, y)

    def full_sensor(self) -> None:
        """Reset ROI to the full sensor extents."""
        log.debug("full_sensor()")
        self.cam.full_sensor()

    def set_timing(self, fps: float | None, exp_ms: float | None) -> None:
        """Set frame rate (fps) and/or exposure (milliseconds)."""
        log.debug("set_timing(fps=%s, exp_ms=%s)", fps, exp_ms)
        self.cam.set_timing(fps, exp_ms)

    def set_gains(self, analog: float | None, digital: float | None) -> None:
        """Set analog and/or digital gains (None -> leave unchanged)."""
        log.debug("set_gains(analog=%s, digital=%s)", analog, digital)
        self.cam.set_gains(analog, digital)

    def refresh_gains(self) -> None:
        """Query current gain values and ranges."""
        log.debug("refresh_gains()")
        self.cam.refresh_gains()

    def refresh_timing(self) -> None:
        """Query current timing (fps/exposure)."""
        log.debug("refresh_timing()")
        if hasattr(self.cam, "refresh_timing"):
            self.cam.refresh_timing()

    def refresh_roi(self) -> None:
        """Query current ROI; only if the facade exposes it."""
        log.debug("refresh_roi()")
        if hasattr(self.cam, "refresh_roi"):
            self.cam.refresh_roi()

    # ---------- utilities -------------------------------------------------
    def desaturate(self, target_frac: float = 0.85, max_iters: int = 5) -> None:
        """
        Run the auto-desaturation loop on the camera until the brightest
        pixels settle below target_frac of full-scale (0..1).
        """
        log.info("desaturate(target=%.3f, iters=%d)", target_frac, max_iters)
        self.cam.auto_desaturate(target_frac, max_iters)

    def shutdown(self) -> None:
        """Best-effort orderly shutdown (used by the main window)."""
        log.info("shutdown()")
        try:
            self.stop()
        except Exception:
            log.exception("stop() during shutdown failed")
        try:
            self.close()
        except Exception:
            log.exception("close() during shutdown failed")
