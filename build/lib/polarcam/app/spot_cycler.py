# src/polarcam/app/spot_cycler.py
from __future__ import annotations
import json, math, os, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Qt

from polarcam.hardware import (
    SENSOR_W, SENSOR_H, STEP_W, STEP_H, MIN_W, MIN_H, MAX_W, MAX_H,
    OFFX_MIN, OFFX_MAX, OFFY_MIN, OFFY_MAX, OFFX_STEP, OFFY_STEP,
    MOSAIC_LAYOUT, MOSAIC_LAYOUT_STR, snap_down,
    roi_for_spot,
)

@dataclass
class CycleConfig:
    out_dir: str
    base_name: str = "cycle"
    dwell_sec: float = 1.0
    max_duration_sec: float = 3600
    chunk_len: int = 20000
    maximize_camera_fps: bool = True
    reassert_timing_each_hop: bool = True
    save_enabled: bool = True
    save_raw: bool = False

class MultiSpotCycler(QObject):
    """
    Cycle a tiny HW ROI across spots and save 4×pol mean signals of a software crop.
    All control calls are queued to the Controller/camera thread; no cross-thread timers.
    """
    # lifecycle / UI
    started       = Signal()
    stopped       = Signal()
    error         = Signal(str)
    progress      = Signal(str)
    advise_ui_cap = Signal(float)   # 20.0 while running, 0.0 when done

    # bridge commands → dev/dev.cam (in their thread)
    _req_set_roi        = Signal(int, int, int, int)  # w,h,x,y
    _req_set_timing     = Signal(object, object)      # fps, exposure_ms
    _req_start_acq      = Signal()
    _req_refresh_roi    = Signal()
    _req_refresh_timing = Signal()

    def __init__(self, dev, spots, cfg: CycleConfig) -> None:
        super().__init__()
        self.dev = dev              # Controller or camera-like
        self.cam = getattr(dev, "cam", None) or dev   # whichever actually emits signals
        self.spots = list(spots or [])
        self.cfg = cfg
        self._want_stop = False

        # snapshots updated via signals
        self._applied_roi = (0, 0, 0, 0)  # x,y,w,h
        self._fps_now = 20.0

        # accumulators
        self._acc_t: list[float] = []
        self._acc0: list[float] = []
        self._acc45: list[float] = []
        self._acc90: list[float] = []
        self._acc135: list[float] = []
        self._chunk_idx: dict[int, int] = {}

        # raw pixel accumulators (separate from means so mid-dwell chunk flush doesn't clear them)
        self._raw_crops: list[np.ndarray] = []
        self._raw_t: list[float] = []

        # saved state
        self._saved_roi: Optional[tuple[int,int,int,int]] = None
        self._saved_fps: Optional[float] = None

        # --- wire command bridge to safest target (Controller preferred) ---
        target = self.dev if hasattr(self.dev, "set_roi") else (self.cam if hasattr(self.cam, "set_roi") else None)
        if target is None:
            raise RuntimeError("Cycler cannot find a target with set_roi/set_timing.")
        self._req_set_roi.connect(target.set_roi, Qt.QueuedConnection)
        if hasattr(target, "set_timing"):  self._req_set_timing.connect(target.set_timing, Qt.QueuedConnection)
        if hasattr(target, "start"):       self._req_start_acq.connect(target.start, Qt.QueuedConnection)
        if hasattr(target, "refresh_roi"): self._req_refresh_roi.connect(target.refresh_roi, Qt.QueuedConnection)
        if hasattr(target, "refresh_timing"): self._req_refresh_timing.connect(target.refresh_timing, Qt.QueuedConnection)

    # ---------- public control ----------
    def start(self) -> None:
        try:
            if not self.spots:
                raise RuntimeError("No spots to cycle.")

            self._setup_dirs()
            self._connect_signals()
            self._save_pre_state()

            if self.cfg.maximize_camera_fps:
                try: self._req_set_timing.emit(float("inf"), None)
                except Exception: pass

            self._want_stop = False
            self.advise_ui_cap.emit(20.0)  # cap preview while cycling
            self.started.emit()
            self._run_loop()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try: self._restore_pre_state()
            except Exception: pass
            self._disconnect_signals()
            self.stopped.emit()
            self.advise_ui_cap.emit(0.0)   # uncap preview

    def stop(self) -> None:
        self._want_stop = True

    # ---------- internals ----------
    def _setup_dirs(self) -> None:
        if not self.cfg.save_enabled and not self.cfg.save_raw:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        for i in range(1, len(self.spots)+1):
            os.makedirs(self._spot_dir(i), exist_ok=True)

    def _spot_dir(self, i: int) -> str:
        return os.path.join(self.cfg.out_dir, f"{self.cfg.base_name}_spot{i:02d}")

    def _save_pre_state(self) -> None:
        self._saved_roi = (0, 0, 0, 0)
        self._saved_fps = None
        try: self._req_refresh_roi.emit()
        except Exception: pass
        try: self._req_refresh_timing.emit()
        except Exception: pass

    def _restore_pre_state(self) -> None:
        try:
            x, y, w, h = self._saved_roi or (0, 0, 0, 0)
            if w and h: self._req_set_roi.emit(w, h, x, y)
        except Exception: pass
        try:
            if self._saved_fps is not None:
                self._req_set_timing.emit(self._saved_fps, None)
        except Exception: pass

    def _connect_signals(self) -> None:
        # Listen from whichever actually emits (cam or controller). Use Direct so
        # the slots run in the emitter's thread; they only touch plain Python data.
        src = self.cam
        try: src.roi.connect(self._on_roi_update, Qt.DirectConnection)
        except Exception: pass
        try: src.timing.connect(self._on_timing_update, Qt.DirectConnection)
        except Exception: pass
        try: src.frame.connect(self._on_frame, Qt.DirectConnection)
        except Exception: pass

    def _disconnect_signals(self) -> None:
        src = self.cam
        for sig, slot in ((getattr(src, "roi", None), self._on_roi_update),
                          (getattr(src, "timing", None), self._on_timing_update),
                          (getattr(src, "frame", None), self._on_frame)):
            try:
                if sig: sig.disconnect(slot)
            except Exception:
                pass

    @Slot(dict)
    def _on_roi_update(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h:
                self._applied_roi = (x, y, w, h)
                if (self._saved_roi == (0, 0, 0, 0)) and (self._saved_roi is not None):
                    self._saved_roi = (x, y, w, h)
        except Exception:
            pass

    @Slot(dict)
    def _on_timing_update(self, d: dict) -> None:
        try:
            rf = d.get("resulting_fps") or d.get("fps")
            if rf is not None:
                self._fps_now = float(rf)
                if self._saved_fps is None:
                    self._saved_fps = self._fps_now
        except Exception:
            pass

    # ----- main loop -----
    def _run_loop(self) -> None:
        t0 = time.perf_counter()
        spot_idx = 0
        dwell_idx = 0
        self._chunk_idx = {i: 0 for i in range(len(self.spots))}

        try: self._req_start_acq.emit()
        except Exception: pass

        while not self._want_stop:
            if (time.perf_counter() - t0) >= max(1.0, float(self.cfg.max_duration_sec)):
                self.progress.emit("Max duration reached; stopping.")
                break

            i = spot_idx % len(self.spots)
            s = self.spots[i]
            cx, cy, r = s.cx, s.cy, s.r

            # Apply tiny HW ROI (queued to controller/camera thread)
            w, h, x, y = roi_for_spot(cx, cy, r)
            try: self._req_set_roi.emit(w, h, x, y)
            except Exception as e:
                self.error.emit(f"Failed set_roi for spot {i+1}: {e}")
                return

            # Let ROI settle (~3 frames); DO NOT pump GUI here
            settle = max(0.03, 3.0 / max(1.0, float(self._fps_now)))
            time.sleep(settle)

            if self.cfg.maximize_camera_fps and self.cfg.reassert_timing_each_hop:
                try:
                    # keep exposure unchanged (None) but push FPS to hardware max
                    self.cam.set_timing(float("inf"), None)
                except Exception:
                    pass
                # brief settle again (1–2 frames) so the camera applies the new budget
                time.sleep(max(0.01, 2.0 / max(1.0, float(self._fps_now))))

            # Dwell
            dwell_t0 = time.perf_counter()
            applied = self._applied_roi
            crop_abs = None
            self._reset_acc()
            self._raw_crops.clear()
            self._raw_t.clear()
            self.progress.emit(f"Spot {i+1}/{len(self.spots)} dwell {dwell_idx+1}: ROI={applied} r\u2248{r:.2f}")

            while (time.perf_counter() - dwell_t0) < max(0.05, float(self.cfg.dwell_sec)) and not self._want_stop:
                # _on_frame runs in emitter thread; we just yield
                time.sleep(0.001)
                if crop_abs is None:
                    crop_abs = getattr(self, "_last_crop_abs", None)
                if self.cfg.save_enabled and len(self._acc_t) >= self.cfg.chunk_len:
                    self._flush_chunk(i, dwell_idx, s, applied, crop_abs)

            if self.cfg.save_enabled and len(self._acc_t):
                self._flush_chunk(i, dwell_idx, s, applied, crop_abs)

            if self.cfg.save_raw and len(self._raw_crops):
                self._flush_raw_dwell(i, dwell_idx, s, applied, crop_abs)

            spot_idx += 1
            if spot_idx % len(self.spots) == 0:
                dwell_idx += 1

    # ----- frame path -----
    @Slot(object)
    def _on_frame(self, arr_obj: object) -> None:
        if self._want_stop: return
        a = np.asarray(arr_obj)
        if a.ndim != 2 or a.size == 0: return
        try:
            ax, ay, aw, ah = self._applied_roi
            if aw <= 0 or ah <= 0:
                H, W = a.shape
                ax = ay = 0; aw = W; ah = H

            vcx = ax + aw * 0.5
            vcy = ay + ah * 0.5
            idx = int(np.argmin([(s.cx-vcx)**2 + (s.cy-vcy)**2 for s in self.spots]))
            s = self.spots[idx]
            cx, cy, r = s.cx, s.cy, s.r

            rcx = float(cx) - float(ax)
            rcy = float(cy) - float(ay)

            r_eff = max(4.0, float(r))
            want  = int(max(10, min(aw, ah, math.ceil(2.0 * r_eff + 6.0))))
            side  = want if (want % 2 == 0) else want + 1

            ix = max(0, min(aw - side, int(round(rcx)) - side // 2))
            iy = max(0, min(ah - side, int(round(rcy)) - side // 2))

            H, W = a.shape
            ix = max(0, min(W - 2, ix)); iy = max(0, min(H - 2, iy))
            jx = max(ix + 1, min(W, ix + side)); jy = max(iy + 1, min(H, iy + side))

            crop = a[iy:jy, ix:jx]

            crop_abs = (ax + ix, ay + iy, int(jx - ix), int(jy - iy))
            self._last_crop_abs = crop_abs

            # Circular mask — only use pixels within spot radius
            ch, cw = crop.shape
            ccx = rcx - ix  # crop-local center
            ccy = rcy - iy
            yy, xx = np.ogrid[:ch, :cw]
            circle = ((xx - ccx)**2 + (yy - ccy)**2) <= r_eff**2

            sy0 = ay + iy
            sx0 = ax + ix
            row_mod = ((np.arange(ch) + sy0) % 2)[:, None]
            col_mod = ((np.arange(cw) + sx0) % 2)[None, :]

            def msk(rm, cm):
                m = circle & (row_mod == rm) & (col_mod == cm)
                vals = crop[m]
                return float(vals.mean()) if vals.size else 0.0

            t = time.perf_counter()
            self._acc_t.append(t)
            self._acc0.append(msk(1, 1));   self._acc45.append(msk(0, 1))
            self._acc90.append(msk(0, 0));  self._acc135.append(msk(1, 0))

            if self.cfg.save_raw:
                self._raw_crops.append(crop.copy())
                self._raw_t.append(t)
        except Exception:
            return

    # ----- helpers -----
    def _reset_acc(self) -> None:
        self._acc_t.clear()
        self._acc0.clear(); self._acc45.clear(); self._acc90.clear(); self._acc135.clear()

    def _flush_chunk(self, spot_i: int, dwell_i: int, spot,
                     applied_roi: Tuple[int, int, int, int] | None,
                     crop_abs: Tuple[int, int, int, int] | None) -> None:
        if not self._acc_t: return
        t0 = float(self._acc_t[0])
        t_rel = np.asarray(self._acc_t, dtype=np.float64) - t0
        c0    = np.asarray(self._acc0, dtype=np.float64)
        c45   = np.asarray(self._acc45, dtype=np.float64)
        c90   = np.asarray(self._acc90, dtype=np.float64)
        c135  = np.asarray(self._acc135, dtype=np.float64)

        meta = {
            "spot": {"cx": spot.cx, "cy": spot.cy, "r": spot.r,
                     "label": spot.label,
                     "phi_cov": spot.phi_cov,
                     "std_median_r": spot.std_median_r},
            "applied_roi": {"x": int(applied_roi[0]), "y": int(applied_roi[1]),
                            "w": int(applied_roi[2]), "h": int(applied_roi[3])} if applied_roi else None,
            "crop_abs": {"x": int(crop_abs[0]), "y": int(crop_abs[1]),
                         "w": int(crop_abs[2]), "h": int(crop_abs[3])} if crop_abs else None,
            "t0_perf_counter": t0,
            "notes": "signals are per-frame means over the displayed/software crop only",
        }
        jmeta = json.dumps(meta)

        sd = self._spot_dir(spot_i + 1)
        idx = self._chunk_idx.get(spot_i, 0)
        self._chunk_idx[spot_i] = idx + 1
        fname = os.path.join(sd, f"{self.cfg.base_name}_s{spot_i+1:02d}_d{dwell_i+1:04d}_p{idx:04d}.npz")

        np.savez_compressed(
            fname,
            t=t_rel, c0=c0, c45=c45, c90=c90, c135=c135,
            meta=np.frombuffer(jmeta.encode("utf-8"), dtype=np.uint8),
        )
        self.progress.emit(f"Saved {fname}  ({len(t_rel)} samples)")
        self._reset_acc()

    def _flush_raw_dwell(self, spot_i: int, dwell_i: int, spot,
                         applied_roi: Tuple[int, int, int, int] | None,
                         crop_abs: Tuple[int, int, int, int] | None) -> None:
        if not self._raw_crops:
            return
        # All crops should be same shape; discard any mismatches (e.g. during ROI transition)
        target_shape = self._raw_crops[-1].shape
        valid_idx = [i for i, c in enumerate(self._raw_crops) if c.shape == target_shape]
        if not valid_idx:
            self._raw_crops.clear()
            self._raw_t.clear()
            return

        stack = np.stack([self._raw_crops[i] for i in valid_idx])
        t_arr = np.array([self._raw_t[i] for i in valid_idx], dtype=np.float64)
        t0 = float(t_arr[0])

        meta: dict = {
            "spot": {"cx": spot.cx, "cy": spot.cy, "r": spot.r,
                     "label": spot.label},
            "applied_roi": {"x": int(applied_roi[0]), "y": int(applied_roi[1]),
                            "w": int(applied_roi[2]), "h": int(applied_roi[3])} if applied_roi else None,
            "crop_abs": {"x": int(crop_abs[0]), "y": int(crop_abs[1]),
                         "w": int(crop_abs[2]), "h": int(crop_abs[3])} if crop_abs else None,
            "t0_perf_counter": t0,
            "n_frames": len(valid_idx),
            "n_discarded": len(self._raw_crops) - len(valid_idx),
        }
        meta["layout"] = MOSAIC_LAYOUT_STR
        if crop_abs:
            tl_sy, tl_sx = crop_abs[1], crop_abs[0]
            meta["crop_top_left_sensor_yx"] = [tl_sy, tl_sx]
            meta["crop_top_left_channel"] = MOSAIC_LAYOUT.get((tl_sy % 2, tl_sx % 2), "?")

        sd = self._spot_dir(spot_i + 1)
        base = f"{self.cfg.base_name}_s{spot_i+1:02d}_d{dwell_i+1:04d}_raw"
        fname = os.path.join(sd, base + ".npz")
        jmeta = json.dumps(meta)
        np.savez(fname, frames=stack, t=t_arr - t0,
                 meta=np.frombuffer(jmeta.encode("utf-8"), dtype=np.uint8))

        # Write companion JSON so pixel‑channel layout is easily readable
        json_path = os.path.join(sd, base + ".json")
        try:
            with open(json_path, "w") as jf:
                json.dump(meta, jf, indent=2)
        except Exception:
            pass

        self.progress.emit(
            f"Raw saved {fname}  ({stack.shape[0]} frames, {stack.shape[1]}\u00d7{stack.shape[2]})")
        self._raw_crops.clear()
        self._raw_t.clear()
