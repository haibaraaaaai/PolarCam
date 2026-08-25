# src/polarcam/app/lut_widget.py
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QLinearGradient, QMouseEvent, QWheelEvent, QPaintEvent, QCursor
from PySide6.QtWidgets import QWidget


class HighlightLUTWidget(QWidget):
    """
    Lightweight highlight-LUT control with drag handles.

    Interactions
    ------------
    • Drag LEFT handle  → black floor (12-bit, 0..4095)
    • Drag RIGHT handle → white cap   (12-bit, 0..4095)
    • Hold CTRL + mouse wheel → adjust gamma
    • Double-click → reset to defaults (floor=0, cap=4095, gamma=1.0)

    Signals
    -------
    paramsChanged(int floor, int cap, float gamma)

    API
    ---
    setParams(floor, cap, gamma)
    params() -> (floor, cap, gamma)
    setHistogram256(hist)  # optional 256 bins over 0..4095
    build_lut() -> np.ndarray shape (4096,), dtype uint8
    """

    paramsChanged = Signal(int, int, float)  # floor, cap, gamma

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # so wheel works when focused

        # state
        self._floor = 0         # 12-bit
        self._cap   = 4095      # 12-bit
        self._gamma = 1.0
        self._drag: Optional[str] = None     # "black" | "white" | None
        self._hist: Optional[np.ndarray] = None  # optional 256-bin histogram

        # visuals / behavior
        self._pad_px = 10
        self._handle_snap_px = 8
        self._bar_top = 14
        self._bar_h = 16
        self._hist_h = 24
        self._gamma_wheel_scale = 0.05  # per wheel step

        self.setToolTip("Drag handles to set floor/cap. Hold Ctrl+Wheel for gamma. Double-click to reset.")

    # ---------------- public API ----------------
    def setParams(self, floor: int, cap: int, gamma: float) -> None:
        f = int(max(0, min(floor, 4094)))
        c = int(max(f + 1, min(cap, 4095)))
        g = float(max(0.05, min(gamma, 10.0)))
        changed = (f != self._floor) or (c != self._cap) or (abs(g - self._gamma) > 1e-9)
        self._floor, self._cap, self._gamma = f, c, g
        if changed:
            self.paramsChanged.emit(self._floor, self._cap, self._gamma)
            self.update()

    def params(self) -> tuple[int, int, float]:
        return self._floor, self._cap, self._gamma

    def setHistogram256(self, hist: Optional[Iterable[float]]) -> None:
        """Set an optional 256-bin histogram for the 12-bit domain (0..4095)."""
        if hist is None:
            self._hist = None
        else:
            arr = np.asarray(list(hist), dtype=np.float32)
            if arr.ndim == 1 and arr.size == 256:
                self._hist = arr
            else:
                # ignore silently if shape is wrong
                self._hist = None
        self.update()

    def build_lut(self) -> np.ndarray:
        """
        Build a 4096-entry uint8 LUT mapping 12-bit values to 8-bit for display:
            <= floor -> 0
            >= cap   -> 255
            else     -> round(255 * ((x - floor) / (cap - floor)) ** gamma)
        """
        x = np.arange(4096, dtype=np.float32)
        f, c, g = float(self._floor), float(self._cap), float(self._gamma)
        denom = max(1.0, c - f)
        t = np.clip((x - f) / denom, 0.0, 1.0)
        y = np.power(t, g) * 255.0
        return y.astype(np.uint8)

    # ---------------- interaction ----------------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        x = e.position().x()
        bx = self._x_for_value(self._floor)
        wx = self._x_for_value(self._cap)
        if abs(x - bx) < self._handle_snap_px:
            self._drag = "black"; self.setCursor(QCursor(Qt.SizeHorCursor))
        elif abs(x - wx) < self._handle_snap_px:
            self._drag = "white"; self.setCursor(QCursor(Qt.SizeHorCursor))
        else:
            self._drag = None
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if not self._drag:
            # hover cursor hint if near a handle
            x = e.position().x()
            bx = self._x_for_value(self._floor)
            wx = self._x_for_value(self._cap)
            if abs(x - bx) < self._handle_snap_px or abs(x - wx) < self._handle_snap_px:
                self.setCursor(QCursor(Qt.SizeHorCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))
            return

        v = self._value_for_x(e.position().x())
        if self._drag == "black":
            new_floor = max(0, min(v, self._cap - 1))
            if new_floor != self._floor:
                self._floor = new_floor
                self.paramsChanged.emit(self._floor, self._cap, self._gamma)
                self.update()
        else:
            new_cap = max(self._floor + 1, min(v, 4095))
            if new_cap != self._cap:
                self._cap = new_cap
                self.paramsChanged.emit(self._floor, self._cap, self._gamma)
                self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._drag = None
        self.setCursor(QCursor(Qt.ArrowCursor))
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        # reset to defaults
        self._floor, self._cap, self._gamma = 0, 4095, 1.0
        self.paramsChanged.emit(self._floor, self._cap, self._gamma)
        self.update()
        super().mouseDoubleClickEvent(e)

    def wheelEvent(self, e: QWheelEvent) -> None:
        # Ctrl+wheel to adjust gamma smoothly; support high-res touchpads.
        if e.modifiers() & Qt.ControlModifier:
            delta = e.angleDelta().y()
            if not delta and e.pixelDelta().y():
                delta = e.pixelDelta().y() / 15.0  # rough scale to "steps"
            delta_steps = delta / 120.0
            g = self._gamma * (1.0 + float(delta_steps) * self._gamma_wheel_scale)
            self._gamma = float(max(0.05, min(g, 10.0)))
            self.paramsChanged.emit(self._floor, self._cap, self._gamma)
            self.update()
            e.accept()
            return
        super().wheelEvent(e)

    # ---------------- drawing helpers ----------------
    def _x_for_value(self, v: int) -> float:
        pad = self._pad_px
        w = max(1, self.width() - 2 * pad)
        return pad + (v / 4095.0) * w

    def _value_for_x(self, x: float) -> int:
        pad = self._pad_px
        w = max(1, self.width() - 2 * pad)
        t = (x - pad) / w
        t = max(0.0, min(1.0, t))
        return int(round(t * 4095))

    # ---------------- paint ----------------
    def paintEvent(self, _: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # background
        p.fillRect(self.rect(), Qt.black)

        pad = self._pad_px
        bar_rect = QRectF(pad, self._bar_top, self.width() - 2 * pad, self._bar_h)

        # histogram backdrop (light gray bars)
        if self._hist is not None and self._hist.size == 256:
            mx = float(np.max(self._hist)) if self._hist.size else 1.0
            mx = max(mx, 1e-9)
            bw = bar_rect.width() / 256.0
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(Qt.darkGray))
            top = bar_rect.bottom()
            for i, v in enumerate(self._hist):
                h = (float(v) / mx) * self._hist_h
                p.drawRect(bar_rect.left() + i * bw, top - h, bw, h)

        # gradient bar
        grad = QLinearGradient(bar_rect.left(), 0, bar_rect.right(), 0)
        grad.setColorAt(0.0, Qt.black)
        grad.setColorAt(1.0, Qt.white)
        p.fillRect(bar_rect, grad)

        # dim the clipped regions (outside floor..cap)
        bx = self._x_for_value(self._floor)
        wx = self._x_for_value(self._cap)
        dim = QBrush(Qt.black)
        p.setPen(Qt.NoPen)
        p.setOpacity(0.4)
        p.fillRect(bar_rect.left(), bar_rect.top(), max(0.0, bx - bar_rect.left()), bar_rect.height(), dim)
        p.fillRect(wx, bar_rect.top(), max(0.0, bar_rect.right() - wx), bar_rect.height(), dim)
        p.setOpacity(1.0)

        # handles
        p.setPen(QPen(Qt.white, 2))
        p.drawLine(int(bx), int(bar_rect.top() - 6), int(bx), int(bar_rect.bottom() + 6))
        p.drawLine(int(wx), int(bar_rect.top() - 6), int(wx), int(bar_rect.bottom() + 6))

        # readout text
        p.setPen(Qt.white)
        p.drawText(
            8,
            self.height() - 8,
            f"floor={self._floor}  cap={self._cap}  gamma={self._gamma:.2f}  (Ctrl+Wheel adjusts gamma; double-click resets)",
        )

        p.end()
