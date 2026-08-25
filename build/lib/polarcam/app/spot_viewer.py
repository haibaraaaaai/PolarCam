# src/polarcam/app/spot_viewer.py
from __future__ import annotations
import math, os, time
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QStatusBar
from PySide6.QtGui import QImage, QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from polarcam.hardware import (
    SENSOR_W, SENSOR_H, STEP_W, STEP_H, MIN_W, MIN_H, snap_down,
    Spot, roi_for_spot,
)

UI_CAP_MAX = 20.0  # preview cap (Hz)
SCATTER_BATCH = 80       # frames per scatter batch
SCATTER_PAUSE_S = 10.0   # seconds between refreshes

# ---- Type-only imports ----
if TYPE_CHECKING:
    from .spot_recorder import SpotSignalRecorder, RecorderConfig  # type: ignore[misc]

# ---- Runtime-optional imports ----
try:
    from .spot_recorder import (
        SpotSignalRecorder as SpotSignalRecorderRT,
        RecorderConfig as RecorderConfigRT,
    )
    _RECORDER_AVAILABLE = True
except Exception:
    SpotSignalRecorderRT = None  # type: ignore[assignment]
    RecorderConfigRT = None      # type: ignore[assignment]
    _RECORDER_AVAILABLE = False


class SpotViewerWindow(QMainWindow):
    """
    Passive spot viewer (UI preview capped at ≤20 Hz) with optional background
    recorder that runs the camera at max FPS. It tolerates being given either
    a Controller (with .cam) or a camera-like object (with .frame/.roi/.timing).
    """
    rec_progress = Signal(int)

    def __init__(
        self,
        dev,  # Controller or camera-like object
        spots,  # List[DetectedSpot] or List[tuple]
        parent: Optional[QMainWindow] = None,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spot viewer — passive preview (≤20 Hz) + optional recorder")
        self.resize(1100, 650)

        self.dev = dev
        self.spots = list(spots or [])
        self.idx = int(max(0, min(index, max(0, len(self.spots) - 1))))

        # ROI to restore when the viewer closes (W, H, X, Y)
        self._saved_roi = kwargs.get('saved_roi')
        self._zoom_roi_set = False

        # UI
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.video = QLabel("Waiting for frames…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")

        left = QVBoxLayout(); left.addWidget(self.video, 1)
        lw = QWidget(); lw.setLayout(left)

        # Scatter plot (q vs u normalised anisotropy)
        self._scatter_fig = Figure(figsize=(3, 3), facecolor='#1a1a1a')
        self._scatter_fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.92)
        self._scatter_ax = self._scatter_fig.add_subplot(111)
        self._style_scatter_axes()
        self._scatter_canvas = FigureCanvasQTAgg(self._scatter_fig)

        self.btn_start = QPushButton("Start Rec @ max FPS")
        self.btn_stop  = QPushButton("Stop Recording"); self.btn_stop.setEnabled(False)
        self.btn_prev  = QPushButton("◀ Prev")
        self.btn_next  = QPushButton("Next ▶")

        if not _RECORDER_AVAILABLE:
            self.btn_start.setEnabled(False)
            self.btn_start.setText("Recorder unavailable")
            self.status.showMessage("Recorder module not found — preview only.", 5000)

        right = QVBoxLayout()
        right.addWidget(self.btn_start)
        right.addWidget(self.btn_stop)
        right.addSpacing(24)
        nav = QHBoxLayout(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
        right.addLayout(nav)
        right.addStretch(1)
        rw = QWidget(); rw.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(lw, 3)
        root.addWidget(self._scatter_canvas, 2)
        root.addWidget(rw, 1)
        w = QWidget(); w.setLayout(root)
        self.setCentralWidget(w)

        # Signals (support dev or dev.cam)
        cam = getattr(self.dev, "cam", self.dev)
        if hasattr(cam, "roi"):
            try: cam.roi.connect(self._on_roi_update, Qt.QueuedConnection)
            except Exception: pass
        if hasattr(cam, "timing"):
            try: cam.timing.connect(self._on_timing_update, Qt.QueuedConnection)
            except Exception: pass
        if hasattr(cam, "frame"):
            try: cam.frame.connect(self._on_frame_arrived, Qt.QueuedConnection)
            except Exception: pass

        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(+1))
        self.btn_start.clicked.connect(self._start_rec)
        self.btn_stop.clicked.connect(self._stop_rec)

        # State — seed from saved_roi so scatter works even if the
        # camera's roi signal was emitted before the viewer was created
        # (always the case with mock backend).
        sr = kwargs.get('saved_roi')  # (W, H, X, Y)
        if sr and sr[0] > 0 and sr[1] > 0:
            self._applied_roi = (sr[2], sr[3], sr[0], sr[1])  # (x, y, w, h)
        else:
            self._applied_roi = (0, 0, 0, 0)
        self._fps_now = 20.0
        self._ui_cap = UI_CAP_MAX
        self._latest = None  # np.ndarray

        # UI tick timer (decoupled from frame rate)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._ui_tick)
        self._timer.start(int(1000.0 / self._ui_cap))

        # Recorder bits
        self._rec_thread: Optional[QThread] = None
        self._rec: Optional['SpotSignalRecorder'] = None  # type-only name quoted

        # Scatter plot state (anisotropy_x / anisotropy_y)
        self._scatter_buf: list[tuple[float, float]] = []
        self._scatter_collecting = True
        self._scatter_pause_timer = QTimer(self)
        self._scatter_pause_timer.setSingleShot(True)
        self._scatter_pause_timer.timeout.connect(self._scatter_resume)
        self._chan_masks: Optional[dict] = None
        self._scatter_crop: Optional[tuple] = None

        self._update_cap()

    # ---- camera snapshots ----
    @Slot(dict)
    def _on_roi_update(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h:
                self._applied_roi = (x, y, w, h)
                self._chan_masks = None  # force scatter geom recompute
        except Exception:
            pass

    @Slot(dict)
    def _on_timing_update(self, d: dict) -> None:
        try:
            rf = d.get("resulting_fps") or d.get("fps")
            if rf is not None:
                self._fps_now = float(rf)
                self._update_cap()
        except Exception:
            pass

    def _update_cap(self) -> None:
        cap = float(min(UI_CAP_MAX, max(1.0, self._fps_now)))
        self._timer.setInterval(int(1000.0 / cap))
        self._ui_cap = cap

    # ---- scatter plot (anisotropy) ----
    # Polar mosaic: (row%2,col%2) → (0,0)=90° (0,1)=45° (1,0)=135° (1,1)=0°

    def _style_scatter_axes(self) -> None:
        ax = self._scatter_ax
        ax.set_facecolor('#111')
        ax.set_xlabel('anisotropy_x  (I\u2080\u2212I\u2089\u2080)/(I\u2080+I\u2089\u2080)', color='#aaa', fontsize=8)
        ax.set_ylabel('anisotropy_y  (I\u2084\u2085\u2212I\u2081\u2083\u2085)/(I\u2084\u2085+I\u2081\u2083\u2085)', color='#aaa', fontsize=8)
        ax.tick_params(colors='#666', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axhline(0, color='#333', lw=0.5)
        ax.axvline(0, color='#333', lw=0.5)

    def _recompute_scatter_geom(self) -> None:
        """Pre-compute crop bounds and per-channel boolean masks for scatter."""
        if not self.spots:
            self._chan_masks = None
            return
        s = self.spots[self.idx]
        cx, cy, r = s.cx, s.cy, s.r
        ax, ay, aw, ah = self._applied_roi
        if aw <= 0 or ah <= 0:
            # Fallback: infer from latest frame (same as _ui_tick)
            if self._latest is not None and self._latest.ndim == 2:
                ah, aw = self._latest.shape
                ax = ay = 0
            else:
                self._chan_masks = None
                return
        rcx = float(cx) - float(ax)
        rcy = float(cy) - float(ay)
        r_eff = max(4.0, float(r))
        want = int(max(10, min(aw, ah, math.ceil(2.0 * r_eff + 6))))
        side = want if (want % 2 == 0) else want + 1
        ix = max(0, min(aw - side, int(round(rcx)) - side // 2))
        iy = max(0, min(ah - side, int(round(rcy)) - side // 2))
        jx = min(aw, ix + side)
        jy = min(ah, iy + side)
        ch, cw = jy - iy, jx - ix
        if ch < 2 or cw < 2:
            self._chan_masks = None
            return
        # Circular mask
        yy, xx = np.ogrid[:ch, :cw]
        dist2 = (xx - (rcx - ix))**2 + (yy - (rcy - iy))**2
        circle = dist2 <= r_eff**2
        # Channel masks from sensor-coordinate mosaic pattern
        sy0, sx0 = ay + iy, ax + ix
        row_mod = ((np.arange(ch) + sy0) % 2)[:, None]
        col_mod = ((np.arange(cw) + sx0) % 2)[None, :]
        self._chan_masks = {
            0:   circle & (row_mod == 1) & (col_mod == 1),
            45:  circle & (row_mod == 0) & (col_mod == 1),
            90:  circle & (row_mod == 0) & (col_mod == 0),
            135: circle & (row_mod == 1) & (col_mod == 0),
        }
        self._scatter_crop = (ix, iy, jx, jy)

    def _scatter_accumulate(self, a: np.ndarray) -> None:
        if self._chan_masks is None:
            self._recompute_scatter_geom()
            if self._chan_masks is None:
                return
        ix, iy, jx, jy = self._scatter_crop  # type: ignore[misc]
        H, W = a.shape
        iy2, jy2 = min(iy, H - 1), min(jy, H)
        ix2, jx2 = min(ix, W - 1), min(jx, W)
        if jy2 <= iy2 or jx2 <= ix2:
            return
        crop = a[iy2:jy2, ix2:jx2]
        masks = self._chan_masks
        if crop.shape != next(iter(masks.values())).shape:
            self._chan_masks = None
            return
        I0   = int(np.sum(crop[masks[0]],   dtype=np.int64))
        I90  = int(np.sum(crop[masks[90]],  dtype=np.int64))
        I45  = int(np.sum(crop[masks[45]],  dtype=np.int64))
        I135 = int(np.sum(crop[masks[135]], dtype=np.int64))
        dq = I0 + I90
        du = I45 + I135
        if dq == 0 or du == 0:
            return
        self._scatter_buf.append(((I0 - I90) / dq, (I45 - I135) / du))
        if len(self._scatter_buf) >= SCATTER_BATCH:
            self._scatter_collecting = False
            self._update_scatter()
            self._scatter_pause_timer.start(int(SCATTER_PAUSE_S * 1000))

    def _update_scatter(self) -> None:
        ax = self._scatter_ax
        ax.clear()
        self._style_scatter_axes()
        if self._scatter_buf:
            qs, us = zip(*self._scatter_buf)
            ax.scatter(qs, us, s=12, c='cyan', alpha=0.7, edgecolors='none')
        self._scatter_canvas.draw_idle()

    def _scatter_resume(self) -> None:
        self._scatter_buf.clear()
        self._scatter_collecting = True

    # ---- frame handling ----
    @Slot(object)
    def _on_frame_arrived(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim == 2:
            self._latest = a
            if self._scatter_collecting:
                self._scatter_accumulate(a)

    def _ui_tick(self) -> None:
        a = self._latest
        if a is None or a.ndim != 2 or not self.spots:
            return

        s = self.spots[self.idx]
        cx, cy, r = s.cx, s.cy, s.r

        ax, ay, aw, ah = self._applied_roi
        if aw <= 0 or ah <= 0:
            ah, aw = a.shape; ax = ay = 0

        rcx = float(cx) - float(ax)
        rcy = float(cy) - float(ay)

        r_eff = max(4.0, float(r))
        want = int(max(10, min(aw, ah, math.ceil(2.0 * r_eff + 6))))
        side = want if (want % 2 == 0) else want + 1

        ix = max(0, min(aw - side, int(round(rcx)) - side // 2))
        iy = max(0, min(ah - side, int(round(rcy)) - side // 2))

        H, W = a.shape
        ix = max(0, min(W - 2, ix)); iy = max(0, min(H - 2, iy))
        jx = max(ix + 1, min(W, ix + side))
        jy = max(iy + 1, min(H, iy + side))

        crop = a[iy:jy, ix:jx]

        # Circular mask: only show pixels within spot radius
        ch, cw = crop.shape
        yy, xx = np.ogrid[:ch, :cw]
        dist2 = (xx - (rcx - ix)) ** 2 + (yy - (rcy - iy)) ** 2
        mask = dist2 <= r_eff ** 2
        masked = crop.copy()
        masked[~mask] = 0

        a8 = (masked >> 4).astype(np.uint8) if masked.dtype == np.uint16 else masked.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8).copy()

        # Scale up with nearest-neighbour so individual pixels are visible
        lbl_w = max(64, self.video.width())
        lbl_h = max(64, self.video.height())
        pm = QPixmap.fromImage(qimg).scaled(
            lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.video.setPixmap(pm)

        # After the first masked render, set a hardware ROI for speed.
        # The mask was already shown using the correct (current) coordinates,
        # so subsequent frames from the zoomed ROI will still be handled
        # correctly via _applied_roi updates from the roi signal.
        if not self._zoom_roi_set:
            self._zoom_roi_set = True
            self._set_spot_roi()

    # ---- navigation ----
    def _jump(self, d: int) -> None:
        if not self.spots:
            return
        self.idx = (self.idx + d) % len(self.spots)
        self._zoom_roi_set = False  # re-set ROI for the new spot
        # Reset scatter for the new spot
        self._chan_masks = None
        self._scatter_buf.clear()
        self._scatter_pause_timer.stop()
        self._scatter_collecting = True

    # ---- hardware ROI for speed ----

    def _set_spot_roi(self) -> None:
        """Set a hardware ROI centred on the current spot to reduce bus bandwidth."""
        if not self.spots:
            return
        s = self.spots[self.idx]
        w, h, x, y = roi_for_spot(s.cx, s.cy, s.r, margin=6, min_r=4.0)
        try:
            if hasattr(self.dev, 'set_roi'):
                self.dev.set_roi(w, h, x, y)
        except Exception:
            pass

    # ---- recorder control ----
    def _start_rec(self) -> None:
        if not _RECORDER_AVAILABLE:
            return
        if not self.spots:
            self.status.showMessage("No spots available (run Detect first).", 2500)
            return

        cam_for_rec = getattr(self.dev, "cam", None)
        if cam_for_rec is None or not (hasattr(cam_for_rec, "roi") and hasattr(cam_for_rec, "frame")):
            cam_for_rec = self.dev

        if not (hasattr(cam_for_rec, "roi") and hasattr(cam_for_rec, "frame")):
            self.status.showMessage("Recorder can’t find a camera with .roi/.frame; preview only.", 4000)
            return

        s = self.spots[self.idx]
        spot = s  # already a Spot instance

        out_dir = os.path.join(os.getcwd(), "captures",
                              time.strftime(f"spot{self.idx+1}_%Y%m%d_%H%M%S"))
        cfg = RecorderConfigRT(
            out_dir=out_dir,
            base_name=f"spot{self.idx+1}",
            chunk_len=20000,
            maximize_camera_fps=True,
        )

        self._rec_thread = QThread(self)
        self._rec = SpotSignalRecorderRT(cam_for_rec, self.dev, spot, cfg)  # runtime class
        self._rec.moveToThread(self._rec_thread)

        self._rec_thread.started.connect(self._rec.start)
        self._rec.progress.connect(self._rec_on_progress, Qt.QueuedConnection)
        self._rec.error.connect(self._rec_on_error, Qt.QueuedConnection)
        self._rec.stopped.connect(self._rec_thread.quit)

        self._rec_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # Pause scatter during recording to avoid any main-thread overhead
        self._scatter_collecting = False
        self._scatter_pause_timer.stop()
        self.status.showMessage("Recording signals at max FPS (preview capped).", 3000)

    def _stop_rec(self) -> None:
        if not _RECORDER_AVAILABLE:
            return
        if self._rec:
            try: self._rec.stop()
            except Exception: pass
        if self._rec_thread:
            self._rec_thread.wait(3000)
        self._rec = None
        self._rec_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Resume scatter
        self._scatter_buf.clear()
        self._scatter_collecting = True
        self.status.showMessage("Recorder stopped.", 2000)

    @Slot(int)
    def _rec_on_progress(self, n: int) -> None:
        self.status.showMessage(f"Recording… {n} samples", 1000)

    @Slot(str)
    def _rec_on_error(self, msg: str) -> None:
        self.status.showMessage(f"Recorder error: {msg}", 4000)

    # ---- cleanup ----
    def closeEvent(self, e):
        try: self._timer.stop()
        except Exception: pass
        try: self._scatter_pause_timer.stop()
        except Exception: pass

        # Restore the hardware ROI that was active before the viewer opened
        if self._saved_roi is not None:
            W, H, X, Y = self._saved_roi
            try:
                if hasattr(self.dev, 'set_roi') and W > 0 and H > 0:
                    self.dev.set_roi(W, H, X, Y)
            except Exception:
                pass

        cam = getattr(self.dev, "cam", self.dev)
        try:
            if hasattr(cam, "frame"):
                cam.frame.disconnect(self._on_frame_arrived)
        except Exception: pass
        try:
            if hasattr(cam, "roi"):
                cam.roi.disconnect(self._on_roi_update)
        except Exception: pass
        try:
            if hasattr(cam, "timing"):
                cam.timing.disconnect(self._on_timing_update)
        except Exception: pass
        try: self._stop_rec()
        except Exception: pass
        super().closeEvent(e)

# Back-compat alias
SpotViewerDialog = SpotViewerWindow
__all__ = ["SpotViewerWindow", "SpotViewerDialog"]
