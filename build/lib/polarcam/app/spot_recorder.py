from __future__ import annotations
import os, json, math, time, threading
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt

from polarcam.hardware import (
    SENSOR_W, SENSOR_H, STEP_W, STEP_H, MIN_W, MIN_H, MAX_W, MAX_H,
    OFFX_MIN, OFFX_MAX, OFFY_MIN, OFFY_MAX, OFFX_STEP, OFFY_STEP,
    MOSAIC_LAYOUT, MOSAIC_LAYOUT_STR, snap_down, clamp, snap_even,
    Spot, roi_for_spot,
)

PAD_SW   = 2      # software padding around diameter
MIN_CROP = 10     # minimum crop side (even)

DEFAULT_CHUNK = 20000  # samples per shard

# --------------------------
# dataclasses
# --------------------------

@dataclass
class RecorderConfig:
    out_dir: str
    base_name: str = "spot"
    chunk_len: int = DEFAULT_CHUNK
    maximize_camera_fps: bool = True   # set_timing(inf, None)

# --------------------------
# shard writer (own thread)
# --------------------------
class _ShardWriter(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, out_dir: str, prefix: str):
        super().__init__()
        self._out_dir = out_dir
        self._prefix  = prefix
        self._q: Deque[np.ndarray] = deque()
        self._cond = threading.Condition()
        self._closing = False
        self._idx = 0

    @Slot(object)
    def enqueue(self, arr: np.ndarray) -> None:
        with self._cond:
            self._q.append(arr)
            self._cond.notify()

    @Slot()
    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify()

    @Slot()
    def run(self) -> None:
        try:
            os.makedirs(self._out_dir, exist_ok=True)
            while True:
                with self._cond:
                    while not self._q and not self._closing:
                        self._cond.wait()
                    if not self._q and self._closing:
                        break
                    chunk = self._q.popleft()
                fn = os.path.join(self._out_dir, f"{self._prefix}_chunk{self._idx:04d}.npy")
                np.save(fn, chunk)   # (N, cropH, cropW) uint16
                self._idx += 1
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# --------------------------
# headless recorder
# --------------------------
class SpotSignalRecorder(QObject):
    started = Signal()
    stopped = Signal()
    error = Signal(str)
    progress = Signal(int)   # total frames written

    def __init__(self, cam, ctrl, spot: Spot, cfg: RecorderConfig):
        super().__init__()
        self.cam  = cam
        self.ctrl = ctrl
        self.spot = spot
        self.cfg  = cfg

        self._applied_roi: Tuple[int,int,int,int] = (0,0,0,0)  # x,y,w,h

        self._chunk_len = int(max(100, cfg.chunk_len))
        self._buf: list[np.ndarray] = []   # list of (cropH, cropW) uint16 frames
        self._ts_buf: list[float] = []     # per-frame timestamps
        self._total = 0

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._prefix = f"{cfg.base_name}_{ts}"
        self._writer = _ShardWriter(cfg.out_dir, self._prefix)
        self._writer_thr = QThread(self)
        self._writer.moveToThread(self._writer_thr)
        self._writer_thr.started.connect(self._writer.run)
        self._writer.finished.connect(self._writer_thr.quit)
        self._writer.error.connect(self.error)

        self.cam.roi.connect(self._on_roi, Qt.QueuedConnection)
        self.cam.frame.connect(self._on_frame, Qt.QueuedConnection)

        self._running = False
        self._t0 = 0.0

        # Pre-computed crop bounds & mask (set once in start())
        self._crop_ix = 0
        self._crop_iy = 0
        self._crop_jx = 0
        self._crop_jy = 0
        self._mask: Optional[np.ndarray] = None  # (cropH, cropW) bool

    # ---- ROI helper (aggressive: H≈2r, W=256) ----
    def _roi_for_spot(self, cx: float, cy: float, r: float) -> tuple[int,int,int,int]:
        return roi_for_spot(cx, cy, r)

    def _compute_crop_and_mask(self, ax: int, ay: int, aw: int, ah: int) -> None:
        """Pre-compute crop bounds and circular mask for _on_frame (called once)."""
        cx, cy = self.spot.cx, self.spot.cy
        r = float(max(1.0, self.spot.r))

        rcx = float(cx) - float(ax)
        rcy = float(cy) - float(ay)

        diameter = max(2.0, 2.0 * r)
        want = int(math.ceil(diameter + 2 * PAD_SW))
        side = snap_even(max(MIN_CROP, min(aw, ah, want)))

        ix = clamp(int(round(rcx)) - side // 2, 0, max(0, aw - side))
        iy = clamp(int(round(rcy)) - side // 2, 0, max(0, ah - side))
        jx = min(aw, ix + side)
        jy = min(ah, iy + side)

        self._crop_ix = ix
        self._crop_iy = iy
        self._crop_jx = jx
        self._crop_jy = jy

        ch = jy - iy
        cw = jx - ix
        yy, xx = np.ogrid[:ch, :cw]
        dist2 = (xx - (rcx - ix)) ** 2 + (yy - (rcy - iy)) ** 2
        self._mask = dist2 <= (r ** 2)

    # ---- lifecycle ----
    def start(self) -> None:
        try: os.makedirs(self.cfg.out_dir, exist_ok=True)
        except Exception: pass

        cx, cy, r = self.spot.cx, self.spot.cy, float(self.spot.r)
        hw, hh, x0, y0 = self._roi_for_spot(cx, cy, r)

        # Pre-set the applied ROI so the crop/mask is correct from
        # the very first frame, before the async roi signal arrives.
        self._applied_roi = (x0, y0, hw, hh)

        # Pre-compute crop bounds and circular mask
        self._compute_crop_and_mask(x0, y0, hw, hh)

        self.ctrl.set_roi(hw, hh, x0, y0)

        if self.cfg.maximize_camera_fps:
            try: self.ctrl.set_timing(float("inf"), None)
            except Exception: pass

        crop_h = self._crop_jy - self._crop_iy
        crop_w = self._crop_jx - self._crop_ix

        # Reference pixel: absolute sensor coords of crop top-left
        tl_sy = y0 + self._crop_iy
        tl_sx = x0 + self._crop_ix
        tl_ch = MOSAIC_LAYOUT[(tl_sy % 2, tl_sx % 2)]

        meta = {
            "spot": {"cx": cx, "cy": cy, "r": self.spot.r,
                     "label": self.spot.label, "phi_cov": self.spot.phi_cov,
                     "std_median_r": self.spot.std_median_r},
            "layout": MOSAIC_LAYOUT_STR,
            "crop_top_left_sensor_yx": [tl_sy, tl_sx],
            "crop_top_left_channel": tl_ch,
            "roi_hw_requested": {"w": hw, "h": hh, "x": x0, "y": y0},
            "crop_hw": [crop_h, crop_w],
            "crop_offset_in_roi": [self._crop_iy, self._crop_ix],
            "chunk_len": self._chunk_len,
            "format": "raw_pixels",
            "dtype": "uint16",
        }
        try:
            with open(os.path.join(self.cfg.out_dir, f"{self._prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        self._writer_thr.start()
        self._running = True
        self._t0 = time.perf_counter()
        self.started.emit()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # flush last partial chunk
        try:
            if self._buf:
                chunk = np.stack(self._buf, axis=0)
                self._writer.enqueue(chunk)
                self._buf = []
        except Exception:
            pass

        # save per-frame timestamps
        try:
            if self._ts_buf:
                ts_arr = np.array(self._ts_buf, dtype=np.float64)
                np.save(os.path.join(self.cfg.out_dir, f"{self._prefix}_timestamps.npy"), ts_arr)
                self._ts_buf = []
        except Exception:
            pass

        # close writer and wait for its thread to finish
        try:
            self._writer.close()
        except Exception:
            pass
        try:
            self._writer_thr.wait(5000)
        except Exception:
            pass

        self.stopped.emit()

    # ---- camera reports ----
    @Slot(dict)
    def _on_roi(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h:
                self._applied_roi = (x, y, w, h)
                # Recompute crop/mask for the new ROI
                self._compute_crop_and_mask(x, y, w, h)
        except Exception:
            pass

    @Slot(object)
    def _on_frame(self, frame_obj: object) -> None:
        if not self._running: return
        a16 = np.asarray(frame_obj)
        if a16.ndim != 2: return

        ix, iy = self._crop_ix, self._crop_iy
        jx, jy = self._crop_jx, self._crop_jy

        H, W = a16.shape
        # Clamp to actual frame dimensions
        iy2 = min(iy, H - 1); jy2 = min(jy, H)
        ix2 = min(ix, W - 1); jx2 = min(jx, W)
        if jy2 <= iy2 or jx2 <= ix2:
            return

        crop = a16[iy2:jy2, ix2:jx2].copy()

        # Apply pre-computed mask (zero outside the spot radius)
        mask = self._mask
        if mask is not None and mask.shape == crop.shape:
            crop[~mask] = 0

        self._ts_buf.append(time.perf_counter())
        self._buf.append(crop)
        self._total += 1
        self.progress.emit(self._total)

        if len(self._buf) >= self._chunk_len:
            try:
                chunk = np.stack(self._buf, axis=0)  # (N, cropH, cropW) uint16
                self._writer.enqueue(chunk)
            except Exception as e:
                self.error.emit(str(e))
            self._buf = []
            self.progress.emit(self._total)
