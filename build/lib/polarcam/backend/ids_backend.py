# ------------------------------------------
# IDS backend (ids_peak): robust worker thread
# ROI + Timing + Gains mailboxes; buffer pool; Mono12/RAW guard.
# Emits uint16 frames (12-bit data in 0..4095).
# ------------------------------------------
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from threading import Event, Lock
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, QMetaObject, Qt

from .base import ICamera

log = logging.getLogger(__name__)

# ---- IDS imports (guarded so UI won’t explode if lib missing) ----
try:
    import ids_peak.ids_peak as ids
    from ids_peak import ids_peak_ipl_extension as ipl_ext
    _IDS_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # pragma: no cover
    ids = None           # type: ignore
    ipl_ext = None       # type: ignore
    _IDS_IMPORT_ERROR = str(e)


@dataclass
class _State:
    open: bool = False
    acquiring: bool = False


# ---- Buffer policy (tunable) ----
BUFFER_WINDOW_SEC = 0.20                 # ~200 ms of frames
BUFFER_MAX_COUNT = 128                   # cap by count
BUFFER_MAX_BYTES = 512 * 1024 * 1024     # cap by bytes (~512 MiB)


# ==================================================================
# Worker that owns the IDS datastream + node map. Lives in QThread.
# ==================================================================
class _StreamWorker(QObject):
    # Signals → GUI thread
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)   # numpy array (H,W) uint16 0..4095
    error = Signal(str)
    roi = Signal(dict)       # {Width, Height, OffsetX, OffsetY}
    timing = Signal(dict)    # {fps, resulting_fps, exposure_us, ...}
    gains = Signal(dict)     # {'analog':{val,min,max,inc}, 'digital':{...}}
    desaturated = Signal(dict)
    auto_desat_started = Signal()
    auto_desat_finished = Signal()

    def __init__(self, node_map, datastream) -> None:
        super().__init__()
        self._nm = node_map
        self._ds = datastream
        self._announced: List[object] = []
        self._run = Event()
        self._closed = False
        self._desat_busy = False

        # Mailboxes (set from GUI thread; read+clear in worker loop)
        self._roi_lock = Lock()
        self._pending_roi: Optional[Dict[str, Any]] = None

        self._tim_lock = Lock()
        self._pending_timing: Optional[Dict[str, Any]] = None

        self._gain_lock = Lock()
        self._pending_gains: Optional[Dict[str, Any]] = None

        self._desat_lock = Lock()
        self._pending_desat: Optional[Tuple[float, int]] = None  # (target_frac, max_iters)

    # ---------- helpers: generic node access ----------
    def _get_node(self, name: str):
        try:
            return self._nm.FindNode(name)
        except Exception:
            return None

    def _node_value(self, name: str) -> Optional[Union[int, float]]:
        n = self._get_node(name)
        if n is None:
            return None
        # value
        for attr in ("Value", "GetValue"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        # enum entry
        for attr in ("CurrentEntry", "GetCurrentEntry"):
            try:
                e = getattr(n, attr)()
                if hasattr(e, "Value"):
                    return float(e.Value())
            except Exception:
                pass
        return None

    def _node_min(self, name: str) -> Optional[float]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Minimum", "GetMinimum", "Min", "GetMin"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        return None

    def _node_max(self, name: str) -> Optional[float]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Maximum", "GetMaximum", "Max", "GetMax"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        return None

    def _node_inc(self, name: str) -> Optional[float]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Increment", "GetIncrement", "Step", "GetStep"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        return None

    def _snap(self, name: str, value: float) -> float:
        """Clamp & snap a value to node min/max/inc."""
        mn = self._node_min(name)
        mx = self._node_max(name)
        inc = self._node_inc(name)
        v = float(value)
        if mn is not None:
            v = max(v, mn)
        if mx is not None:
            v = min(v, mx)
        if inc and inc > 0:
            base = mn if mn is not None else 0.0
            steps = math.floor((v - base) / inc + 1e-9)
            v = base + steps * inc
        if mx is not None and v > mx:
            v = mx
        # tiny nudge away from hard max (some IDS nodes reject exact max)
        if mx is not None and abs(v - mx) < 1e-7:
            v = mx - (inc or 1e-3) * 1e-3
        return float(v)

    # ---------- buffer pool ----------
    def _current_payload_bytes(self) -> int:
        try:
            return int(self._nm.FindNode("PayloadSize").Value())
        except Exception:
            return 0

    def _estimate_target_buffers(self, desired_fps: Optional[float] = None) -> Tuple[int, int, int]:
        """Return (target_count, required_min, payload_bytes)."""
        try:
            required = int(self._ds.NumBuffersAnnouncedMinRequired())
        except Exception:
            required = 3
        required = max(required, 1)

        # pick FPS estimate
        fps: Optional[float] = None
        if desired_fps is not None:
            try:
                fps = float(desired_fps)
            except Exception:
                fps = None
        if fps is None:
            t = self._read_timing()
            fps = (t.get("resulting_fps") or t.get("fps") or 30.0)  # type: ignore[assignment]
            try:
                fps = float(fps)  # type: ignore[arg-type]
            except Exception:
                fps = 30.0

        payload = self._current_payload_bytes()
        window_count = int(max(1, round(float(fps) * BUFFER_WINDOW_SEC)))
        base = max(8, 2 * required)
        target = max(base, window_count)

        if payload > 0:
            max_by_mem = max(1, int(BUFFER_MAX_BYTES // payload))
            target = min(target, BUFFER_MAX_COUNT, max_by_mem)
        else:
            target = min(target, BUFFER_MAX_COUNT)
        return target, required, payload

    def _clear_pool(self) -> None:
        try:
            self._ds.Flush(ids.DataStreamFlushMode_DiscardAll)
        except Exception:
            pass
        try:
            for b in self._announced:
                try:
                    self._ds.RevokeBuffer(b)
                except Exception:
                    pass
        except Exception:
            pass
        self._announced = []

    def _announce_and_queue(self, desired_fps: Optional[float] = None) -> bool:
        try:
            self._clear_pool()
            target, required, payload = self._estimate_target_buffers(desired_fps)
            if payload <= 0:
                self.error.emit("PayloadSize is 0")
                return False
            for _ in range(target):
                buf = self._ds.AllocAndAnnounceBuffer(payload)
                self._announced.append(buf)
            for b in self._announced:
                self._ds.QueueBuffer(b)
            sleep(0.005)
            try:
                mb = payload * len(self._announced)
                log.debug("BufferPool: announced=%d required>=%d payload=%d (~%.1f MiB)",
                          len(self._announced), required, payload, mb / 1048576.0)
            except Exception:
                pass
            return True
        except Exception as e:
            self.error.emit(f"Pool prepare failed: {e}")
            log.exception("Pool prepare failed")
            return False

    # ---------- pixel format ----------
    def _force_mono_format(self) -> None:
        """Best-effort grayscale output (prefer Mono12/Mono16/Mono8)."""
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            tln and tln.SetValue(0)
            log.debug("TLParamsLocked=0 for PixelFormat")
        except Exception:
            pass
        try:
            pf = self._nm.FindNode("PixelFormat")
            if pf and hasattr(pf, "SetCurrentEntry"):
                for cand in ("Mono12", "Mono16", "Mono8", "Mono12p"):
                    try:
                        pf.SetCurrentEntry(cand)
                        log.info("PixelFormat -> %s", cand)
                        break
                    except Exception:
                        continue
        except Exception:
            pass
        finally:
            try:
                tln = self._nm.FindNode("TLParamsLocked")
                tln and tln.SetValue(1)
                log.debug("TLParamsLocked=1 after PixelFormat")
            except Exception:
                pass

    # ---------- format guard (RAW + Mono12) ----------
    def _enum_symbol(self, node) -> Optional[str]:
        """Best-effort readback of the current enum symbolic name."""
        try:
            ce = getattr(node, "CurrentEntry", None)
            if callable(ce):
                ent = ce()
                for attr in ("SymbolicValue", "GetSymbolic", "Symbolic", "GetSymbolicValue"):
                    fn = getattr(ent, attr, None)
                    if callable(fn):
                        try:
                            val = fn()
                            if isinstance(val, str):
                                return val
                        except Exception:
                            pass
                for attr in ("DisplayName", "GetDisplayName"):
                    fn = getattr(ent, attr, None)
                    if callable(fn):
                        try:
                            val = fn()
                            if isinstance(val, str):
                                return val
                        except Exception:
                            pass
        except Exception:
            pass
        return None

    def _set_enum_symbol(self, node_name: str, symbolic: str) -> bool:
        n = self._nm.FindNode(node_name)
        if n is None:
            return False
        try:
            if hasattr(n, "SetCurrentEntry"):
                n.SetCurrentEntry(symbolic)
                return True
        except Exception:
            pass
        return False

    def _node_bool(self, name: str) -> Optional[bool]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Value", "GetValue"):
            try:
                return bool(getattr(n, attr)())
            except Exception:
                pass
        # Sometimes exposed as enum "True"/"False"
        try:
            sym = self._enum_symbol(n)
            if isinstance(sym, str):
                s = sym.strip().lower()
                if s in ("true", "1"):
                    return True
                if s in ("false", "0"):
                    return False
        except Exception:
            pass
        return None

    @Slot()
    def force_raw_mono12(self) -> None:
        """Enforce Component=Raw + PixelFormat=Mono12 when NOT acquiring.
        Skips writes if already correct (avoid BAD_ACCESS on read-only nodes)."""
        if getattr(self, "_run", None) and self._run.is_set():
            self.error.emit("Format change requires camera to be stopped.")
            return

        log.info("FormatGuard: enforce Component=Raw + Mono12")

        # Unlock while we touch format nodes
        tln = None
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            tln and tln.SetValue(0)
            log.debug("TLParamsLocked=0 for format guard")
        except Exception:
            pass

        try:
            sel = self._get_node("ComponentSelector")
            pf  = self._get_node("PixelFormat")
            if sel is None:
                self.error.emit("ComponentSelector node not found.")
                return
            if pf is None:
                self.error.emit("PixelFormat node not found.")
                return

            curr_comp = (self._enum_symbol(sel) or "").lower()
            en_val    = self._node_bool("ComponentEnable")
            pf_sym    = (self._enum_symbol(pf) or "").lower()

            # Fast path
            if curr_comp == "raw" and pf_sym.startswith("mono12"):
                log.info("FormatGuard: already RAW + Mono12; nothing to do.")
                return

            # 1) Select RAW
            if curr_comp != "raw":
                ok = False
                for cand in ("Raw", "RAW", "raw"):
                    if self._set_enum_symbol("ComponentSelector", cand):
                        new_sym = (self._enum_symbol(sel) or "").lower()
                        if new_sym == "raw":
                            ok = True
                            break
                if not ok:
                    self.error.emit("Could not set ComponentSelector to 'Raw'.")
                    return
                en_val = self._node_bool("ComponentEnable")  # refresh

            # 2) Ensure RAW component enabled (only write if needed)
            en_node = self._get_node("ComponentEnable")
            if en_node is not None:
                if en_val is not True:
                    try:
                        en_node.SetValue(True)
                        log.info("FormatGuard: ComponentEnable -> True")
                    except Exception as e:
                        reread = self._node_bool("ComponentEnable")
                        if reread is True:
                            log.info("FormatGuard: ComponentEnable already active (read-only).")
                        else:
                            self.error.emit(f"Could not enable RAW component: {e}")
                            return
            else:
                log.debug("FormatGuard: ComponentEnable node not present; continuing.")

            # 3) Set PixelFormat to Mono12 (strict, tolerate already-Mono12)
            if not pf_sym.startswith("mono12"):
                try:
                    pf.SetCurrentEntry("Mono12")
                    log.info("PixelFormat -> Mono12")
                except Exception as e_pf:
                    pf_sym2 = (self._enum_symbol(pf) or "").lower()
                    if pf_sym2.startswith("mono12"):
                        log.info("FormatGuard: PixelFormat already Mono12.")
                    else:
                        for fb in ("Mono12p", "Mono12Packed"):
                            try:
                                pf.SetCurrentEntry(fb)
                                log.info("PixelFormat -> %s", fb)
                                break
                            except Exception:
                                continue
                        else:
                            self.error.emit(f"Could not set PixelFormat to Mono12: {e_pf}")
                            return

            log.info("FormatGuard: RAW + Mono12 enforced.")
            # Keep GUI in sync
            self.query_roi()
            self.query_timing()

        finally:
            try:
                tln and tln.SetValue(1)
                log.debug("TLParamsLocked=1 after format guard")
            except Exception:
                pass

    # ---------- ROI mailbox ----------
    def _pop_pending_roi(self) -> Optional[Dict[str, Any]]:
        with self._roi_lock:
            d = self._pending_roi
            self._pending_roi = None
            return d

    def _img_dims(self, img) -> Tuple[int, int]:
        try:
            return int(img.Width()), int(img.Height())
        except Exception:
            pass
        try:
            sz = img.Size()
            w = getattr(sz, "width", None); h = getattr(sz, "height", None)
            if w is not None and h is not None:
                return int(w), int(h)
            W = getattr(sz, "Width", None); H = getattr(sz, "Height", None)
            if callable(W) and callable(H):
                return int(W()), int(H())
            if W is not None and H is not None:
                return int(W), int(H)
        except Exception:
            pass
        raise RuntimeError("Cannot obtain image size from IDS IPL image")

    def _apply_roi_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        try:
            w = float(d.get("Width")); h = float(d.get("Height"))
            x = float(d.get("OffsetX")); y = float(d.get("OffsetY"))
            log.debug("apply_roi while_running=%s: W=%s H=%s X=%s Y=%s",
                      while_running, w, h, x, y)
        except Exception as e:
            self.error.emit(f"ROI args invalid: {e}")
            return

        # pause if needed
        if while_running:
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass
            self._clear_pool()

        # unlock
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            tln and tln.SetValue(0)
            log.debug("TLParamsLocked=0 for ROI")
        except Exception:
            pass

        # set size first (resets offsets to allowed ranges)
        try:
            wv = int(round(self._snap("Width",  w)))
            hv = int(round(self._snap("Height", h)))
            self._nm.FindNode("Width").SetValue(wv)
            self._nm.FindNode("Height").SetValue(hv)
            log.info("ROI size -> Width=%d Height=%d", wv, hv)
        except Exception as e:
            self.error.emit(f"ROI width/height failed: {e}")
            return

        # then offsets
        try:
            xv = int(round(self._snap("OffsetX", x)))
            yv = int(round(self._snap("OffsetY", y)))
            mx_x = self._node_max("OffsetX"); mx_y = self._node_max("OffsetY")
            if mx_x is not None:
                xv = min(xv, int(round(mx_x)))
            if mx_y is not None:
                yv = min(yv, int(round(mx_y)))
            self._nm.FindNode("OffsetX").SetValue(xv)
            self._nm.FindNode("OffsetY").SetValue(yv)
            log.info("ROI offsets -> X=%d Y=%d", xv, yv)
        except Exception as e:
            self.error.emit(f"ROI offsets failed: {e}")
            return

        # re-arm buffers + relock
        self._force_mono_format()
        self._announce_and_queue(desired_fps=None)
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            tln and tln.SetValue(1)
            log.debug("TLParamsLocked=1 after ROI")
        except Exception:
            pass

        # resume if needed
        if while_running:
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                log.debug("resumed acquisition after ROI")
            except Exception as e:
                self.error.emit(f"ROI resume failed: {e}")

        self.query_roi()
        self.query_timing()

    def _apply_full_sensor(self, while_running: bool) -> None:
        if while_running:
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass
            self._clear_pool()

        try:
            tln = self._nm.FindNode("TLParamsLocked")
            tln and tln.SetValue(0)
            # reset offsets to minima
            for key in ("OffsetX", "OffsetY"):
                mn = self._node_min(key)
                if mn is not None:
                    self._nm.FindNode(key).SetValue(int(round(mn)))
            # set max size
            wmx = self._node_max("Width"); hmx = self._node_max("Height")
            if wmx is not None:
                self._nm.FindNode("Width").SetValue(int(round(wmx)))
            if hmx is not None:
                self._nm.FindNode("Height").SetValue(int(round(hmx)))
            tln and tln.SetValue(1)
        except Exception as e:
            self.error.emit(f"Full sensor failed: {e}")
            return

        self._force_mono_format()
        self._announce_and_queue(desired_fps=None)

        if while_running:
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                log.debug("resumed acquisition after ROI(full)")
            except Exception as e:
                self.error.emit(f"ROI resume failed: {e}")

        self.query_roi()
        self.query_timing()

    @Slot()
    def enqueue_full_sensor(self) -> None:
        # Reuse the ROI mailbox with a sentinel key
        with self._roi_lock:
            self._pending_roi = {"__full__": True}

    # ---------- Timing mailbox ----------
    def _read_timing(self) -> Dict[str, Union[float, bool, None]]:
        d: Dict[str, Union[float, bool, None]] = {}
        fps_node = self._get_node("AcquisitionFrameRate")
        d["fps_writable"] = fps_node is not None
        if fps_node is not None:
            d["fps"] = self._node_value("AcquisitionFrameRate")
            d["fps_min"] = self._node_min("AcquisitionFrameRate")
            d["fps_max"] = self._node_max("AcquisitionFrameRate")
            d["fps_inc"] = self._node_inc("AcquisitionFrameRate")
        else:
            d["fps"] = d["fps_min"] = d["fps_max"] = d["fps_inc"] = None
        rf = self._node_value("ResultingFrameRate")
        if rf is not None:
            d["resulting_fps"] = rf
        d["exposure_us"]  = self._node_value("ExposureTime")
        d["exposure_min"] = self._node_min("ExposureTime")
        d["exposure_max"] = self._node_max("ExposureTime")
        d["exposure_inc"] = self._node_inc("ExposureTime")
        return d

    def _apply_timing_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        log.debug("apply_timing while_running=%s: %s", while_running, d)
        fps_req = d.get("fps")
        exp_ms_req = d.get("exposure_ms")

        def _unlock():
            try:
                tln = self._nm.FindNode("TLParamsLocked")
                tln and tln.SetValue(0)
                log.debug("TLParamsLocked=0 for timing")
            except Exception:
                pass

        def _pause():
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass

        def _resume():
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                log.debug("resumed acquisition after timing")
            except Exception as e:
                self.error.emit(f"Timing resume failed: {e}")

        def _rebuild_pool(desired_fps: Optional[float] = None):
            if self._announce_and_queue(desired_fps):
                log.debug("buffer pool rebuilt for timing")

        _unlock()
        paused = False

        # Exposure first (it can change FPS caps)
        def _set_exposure(exp_ms: float):
            nonlocal paused
            exp_us = self._snap("ExposureTime", float(exp_ms) * 1000.0)
            try:
                auto = self._get_node("ExposureAuto")
                if auto and hasattr(auto, "SetCurrentEntry"):
                    try:
                        auto.SetCurrentEntry("Off")
                    except Exception:
                        pass
                self._get_node("ExposureTime").SetValue(exp_us)
                log.info("Exposure -> %.1f us", exp_us)
            except Exception as e:
                if while_running and not paused:
                    _pause(); paused = True
                    try:
                        self._get_node("ExposureTime").SetValue(exp_us)
                        log.info("Exposure (paused) -> %.1f us", exp_us)
                    except Exception as e2:
                        self.error.emit(f"Set Exposure failed: {e2}")

        def _set_fps(fps_val: float):
            nonlocal paused
            fps_snapped = self._snap("AcquisitionFrameRate", float(fps_val))
            try:
                en = self._get_node("AcquisitionFrameRateEnable")
                if en is not None:
                    try:
                        en.SetValue(True)
                    except Exception:
                        pass
                self._get_node("AcquisitionFrameRate").SetValue(fps_snapped)
                log.info("FrameRate -> %.3f", fps_snapped)
            except Exception as e:
                if while_running and not paused:
                    _pause(); paused = True
                    try:
                        self._get_node("AcquisitionFrameRate").SetValue(fps_snapped)
                        log.info("FrameRate (paused) -> %.3f", fps_snapped)
                    except Exception as e2:
                        self.error.emit(f"Set FPS failed: {e2}")

        if exp_ms_req is not None:
            _set_exposure(exp_ms_req)
        if fps_req is not None:
            _set_fps(fps_req)

        # relock + maybe rebuild + resume
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            tln and tln.SetValue(1)
            log.debug("TLParamsLocked=1 after timing")
        except Exception:
            pass
        if paused and while_running:
            self._force_mono_format()
            _rebuild_pool(desired_fps=fps_req if fps_req is not None else None)
            _resume()

        self.query_timing()

    # ---------- Gains mailbox ----------
    def _read_gains(self) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {"analog": {}, "digital": {}}
        try:
            # Analog
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry("AnalogAll")
                    except Exception:
                        pass
                out["analog"]["val"] = self._node_value("Gain")
                out["analog"]["min"] = self._node_min("Gain")
                out["analog"]["max"] = self._node_max("Gain")
                out["analog"]["inc"] = self._node_inc("Gain")
            except Exception:
                out["analog"] = {"val": None, "min": None, "max": None, "inc": None}
            # Digital
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry("DigitalAll")
                    except Exception:
                        pass
                out["digital"]["val"] = self._node_value("Gain")
                out["digital"]["min"] = self._node_min("Gain")
                out["digital"]["max"] = self._node_max("Gain")
                out["digital"]["inc"] = self._node_inc("Gain")
            except Exception:
                out["digital"] = {"val": None, "min": None, "max": None, "inc": None}
        except Exception:
            out = {"analog": {"val": None, "min": None, "max": None, "inc": None},
                   "digital": {"val": None, "min": None, "max": None, "inc": None}}
        return out

    def _apply_gains_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        log.debug("apply_gains while_running=%s: %s", while_running, d)

        def _pause():
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass

        def _resume():
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                log.debug("resumed acquisition after gains")
            except Exception as e:
                self.error.emit(f"Gains resume failed: {e}")

        paused = False
        if while_running:
            _pause(); paused = True

        def _set_gain(sel_name: str, val: float):
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry(sel_name)
                    except Exception:
                        pass
                v = float(val)
                mn, mx, inc = self._node_min("Gain"), self._node_max("Gain"), self._node_inc("Gain")
                if mn is not None:
                    v = max(v, mn)
                if mx is not None:
                    v = min(v, mx)
                if inc and inc > 0:
                    base = mn if mn is not None else 0.0
                    steps = math.floor((v - base) / inc + 1e-9)
                    v = base + steps * inc
                self._get_node("Gain").SetValue(v)
                log.info("%s Gain -> %.3f", sel_name, v)
            except Exception as e:
                self.error.emit(f"Set {sel_name} Gain failed: {e}")

        if "analog" in d and d["analog"] is not None:
            _set_gain("AnalogAll", float(d["analog"]))
        if "digital" in d and d["digital"] is not None:
            _set_gain("DigitalAll", float(d["digital"]))

        if paused and while_running:
            self._force_mono_format()
            if self._announce_and_queue(desired_fps=None):
                log.debug("buffer pool rebuilt for gains")
            _resume()

        self.gains.emit(self._read_gains())

    # ---------- public slots (run in worker thread) ----------
    @Slot()
    def start(self) -> None:
        if self._closed or self._run.is_set():
            return
        try:
            log.debug("worker: start requested")
            self._force_mono_format()
            if not self._announce_and_queue(desired_fps=None):
                self.error.emit("Start failed: buffer pool not ready")
                return
            try:
                n = self._nm.FindNode("TLParamsLocked"); n and n.SetValue(1)
            except Exception:
                pass
            self._ds.StartAcquisition()
            self._nm.FindNode("AcquisitionStart").Execute()
            self._run.set()
            self.started.emit()
            log.info("worker: started acquisition")

            while self._run.is_set():
                # auto-desaturate request
                with self._desat_lock:
                    p = self._pending_desat
                    self._pending_desat = None
                if p is not None:
                    self._run_auto_desat(*p)
                    continue
                # apply any pending mailbox updates (one per loop)
                pending_roi = self._pop_pending_roi()
                if pending_roi is not None:
                    if pending_roi.get("__full__"):
                        self._apply_full_sensor(while_running=True)
                    else:
                        self._apply_roi_payload(pending_roi, while_running=True)
                    continue
                with self._tim_lock:
                    t = self._pending_timing; self._pending_timing = None
                if t is not None:
                    self._apply_timing_payload(t, while_running=True)
                    continue
                with self._gain_lock:
                    g = self._pending_gains; self._pending_gains = None
                if g is not None:
                    if g.get("__refresh__"):
                        self.gains.emit(self._read_gains())
                    else:
                        self._apply_gains_payload(g, while_running=True)
                    continue

                # wait for frame
                buf = None
                try:
                    buf = self._ds.WaitForFinishedBuffer(10)  # ms
                except Exception:
                    continue

                try:
                    img = ipl_ext.BufferToImage(buf)
                    if not hasattr(img, "get_numpy_1D"):
                        raise RuntimeError("IDS IPL image missing numpy accessor")
                    w, h = self._img_dims(img)
                    flat = np.asarray(img.get_numpy_1D())
                    if flat.ndim != 1:
                        flat = flat.reshape(-1)

                    expected = w * h
                    # If buffer is bytes (2*expected), re-interpret little-endian to uint16
                    if flat.size == 2 * expected and flat.dtype == np.uint8:
                        if not flat.flags.c_contiguous:
                            flat = np.ascontiguousarray(flat)
                        arr = flat.view("<u2").reshape(h, w).copy()
                    else:
                        arr = flat.reshape(h, w).copy()

                    # Standardize to uint16, scale 8→12bit if needed
                    if arr.dtype == np.uint8:
                        arr = (arr.astype(np.uint16) << 4)
                    elif arr.dtype != np.uint16:
                        arr = arr.astype(np.uint16, copy=False)

                    self.frame.emit(arr)  # 0..4095 expected
                except Exception as e:
                    self.error.emit(f"Convert failed: {e}")
                finally:
                    if buf is not None:
                        try:
                            self._ds.QueueBuffer(buf)
                        except Exception:
                            pass

            # graceful stop
            log.debug("worker: finalize stop begin")
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass
            self._clear_pool()
            try:
                n = self._nm.FindNode("TLParamsLocked"); n and n.SetValue(0)
            except Exception:
                pass
            log.debug("worker: finalize stop done")
            self.stopped.emit()
        except Exception as e:
            self.error.emit(f"Start failed: {e}")
            log.exception("worker: start exception")

    @Slot()
    def set_stop_flag(self) -> None:
        log.debug("worker: set_stop_flag")
        self._run.clear()

    @Slot()
    def request_stop(self) -> None:
        log.debug("worker: request_stop (CancelWait)")
        self._run.clear()
        try:
            cancel = getattr(self._ds, "CancelWait", None)
            if callable(cancel):
                cancel()
        except Exception:
            pass

    @Slot()
    def process_pending_roi(self) -> None:
        d = self._pop_pending_roi()
        if d is not None:
            if d.get("__full__"):
                self._apply_full_sensor(while_running=False)
            else:
                self._apply_roi_payload(d, while_running=False)

    # ---- ROI API ----
    @Slot()
    def query_roi(self) -> None:
        names = ("Width", "Height", "OffsetX", "OffsetY")
        out: Dict[str, Optional[float]] = {}
        for n in names:
            out[n] = self._node_value(n)
        log.debug("ROI snapshot -> %s", out)
        self.roi.emit(out)

    @Slot(object)
    def enqueue_roi(self, roi: object) -> None:
        log.debug("enqueue_roi <- %s", roi)
        if not isinstance(roi, dict):
            return
        with self._roi_lock:
            self._pending_roi = roi

    # ---- Timing API ----
    @Slot()
    def query_timing(self) -> None:
        d = self._read_timing()
        log.debug("timing -> %s", d)
        self.timing.emit(d)

    @Slot(object)
    def enqueue_timing(self, timing: object) -> None:
        log.debug("enqueue_timing <- %s", timing)
        if not isinstance(timing, dict):
            return
        with self._tim_lock:
            self._pending_timing = timing

    @Slot()
    def process_pending_timing(self) -> None:
        with self._tim_lock:
            d = self._pending_timing
            self._pending_timing = None
        if d is not None:
            self._apply_timing_payload(d, while_running=False)

    # ---- Gains API ----
    @Slot()
    def query_gains(self) -> None:
        g = self._read_gains()
        log.debug("gains -> %s", g)
        self.gains.emit(g)

    @Slot(object)
    def enqueue_gains(self, gains: object) -> None:
        log.debug("enqueue_gains <- %s", gains)
        if not isinstance(gains, dict):
            return
        with self._gain_lock:
            self._pending_gains = gains

    @Slot()
    def process_pending_gains(self) -> None:
        with self._gain_lock:
            g = self._pending_gains
            self._pending_gains = None
        if g is not None:
            if g.get("__refresh__"):
                self.gains.emit(self._read_gains())
                return
            self._apply_gains_payload(g, while_running=False)

    @Slot(float, int)
    def enqueue_auto_desat(self, target_frac: float, max_iters: int) -> None:
        log.debug("enqueue_auto_desat <- target=%s, max_iters=%s", target_frac, max_iters)
        if self._desat_busy:
            log.debug("AutoDesat already running; ignoring")
            return
        self._desat_busy = True
        self.auto_desat_started.emit()
        try:
            self._run_auto_desat(target_frac, max_iters)
        finally:
            self._desat_busy = False
            self.auto_desat_finished.emit()

    def _sample_stats(self, timeout_ms: int = 150) -> Optional[tuple[int, int]]:
        """Return (pMax, p99_9) of a fresh frame, or None on timeout."""
        buf = None
        try:
            buf = self._ds.WaitForFinishedBuffer(timeout_ms)
        except Exception:
            return None
        try:
            img = ipl_ext.BufferToImage(buf)
            w, h = self._img_dims(img)
            flat = np.asarray(img.get_numpy_1D())
            if flat.ndim != 1:
                flat = flat.reshape(-1)
            expected = w * h
            if flat.size == 2 * expected and flat.dtype == np.uint8:
                if not flat.flags.c_contiguous:
                    flat = np.ascontiguousarray(flat)
                arr = flat.view("<u2").reshape(h, w)
            else:
                arr = flat.reshape(h, w)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16, copy=False)
            v = arr.ravel()
            pmax = int(v.max())
            p = np.partition(v, int(0.999 * (v.size - 1)))[int(0.999 * (v.size - 1))]
            p999 = int(p)
            return pmax, p999
        finally:
            if buf is not None:
                try:
                    self._ds.QueueBuffer(buf)
                except Exception:
                    pass

    def _pause_stream(self):
        try:
            self._nm.FindNode("AcquisitionStop").Execute()
        except Exception:
            pass
        try:
            self._ds.StopAcquisition()
        except Exception:
            pass

    def _resume_stream(self):
        try:
            self._ds.StartAcquisition()
            self._nm.FindNode("AcquisitionStart").Execute()
        except Exception as e:
            self.error.emit(f"AutoDesat resume failed: {e}")

    def _run_auto_desat(self, target_frac: float, max_iters: int) -> None:
        target_frac = float(max(0.05, min(target_frac, 0.999)))
        target_dn = int(round(4095.0 * target_frac))
        iters = max(1, int(max_iters))

        log.info("AutoDesat: target=%d (~%.1f%% FS) steps=%d", target_dn, target_frac * 100.0, iters)

        for i in range(1, iters + 1):
            stats = self._sample_stats(timeout_ms=200)
            if stats is None:
                log.debug("AutoDesat: iter %d/%d: no frame (timeout); retrying…", i, iters)
                continue
            pmax, p999 = stats
            peak_for_ctrl = p999 if pmax > target_dn * 1.30 else pmax
            cur_exp = float(self._node_value("ExposureTime") or 10000.0)
            log.debug("AutoDesat: iter %d/%d: pMax=%d p99.9=%d target=%d exp=%.1fus",
                      i, iters, pmax, p999, target_dn, cur_exp)

            if peak_for_ctrl <= target_dn:
                log.info("AutoDesat: done in %d step(s); metric=%d ≤ target=%d", i, peak_for_ctrl, target_dn)
                break

            factor = max(0.05, min(1.0, (target_dn / max(1.0, float(peak_for_ctrl))) * 0.92))
            new_exp = self._snap("ExposureTime", cur_exp * factor)

            self._pause_stream()
            try:
                tln = self._nm.FindNode("TLParamsLocked"); tln and tln.SetValue(0)
            except Exception:
                pass
            try:
                self._get_node("ExposureTime").SetValue(new_exp)
                log.info("AutoDesat: Exposure -> %.1f us (factor=%.3f)", new_exp, factor)
            except Exception as e:
                self.error.emit(f"AutoDesat: set Exposure failed: {e}")
            finally:
                try:
                    tln = self._nm.FindNode("TLParamsLocked"); tln and tln.SetValue(1)
                except Exception:
                    pass

            self._announce_and_queue(desired_fps=None)
            self._resume_stream()

            rf = self._node_value("ResultingFrameRate") or self._node_value("AcquisitionFrameRate") or 30.0
            settle = max(0.06, 3.0 / float(rf))
            sleep(settle)

        else:
            stats = self._sample_stats(timeout_ms=200)
            final = stats[0] if stats else None
            if (final is not None) and (final <= target_dn):
                log.info("AutoDesat: achieved after settle: peak=%d ≤ target=%d", final, target_dn)
            else:
                msg = "AutoDesat: max steps reached; peak still above target."
                log.warning(msg)
                self.error.emit(msg)

        self.query_timing()

    @Slot()
    def process_pending_desat(self) -> None:
        with self._desat_lock:
            p = self._pending_desat
            self._pending_desat = None
        if p is not None:
            self._run_auto_desat(*p)

    @Slot()
    def close(self) -> None:
        if self._closed:
            return
        log.debug("worker: close requested")
        self._run.clear()
        try:
            self._nm.FindNode("AcquisitionStop").Execute()
        except Exception:
            pass
        try:
            self._ds.StopAcquisition()
        except Exception:
            pass
        self._clear_pool()
        self._closed = True
        log.info("worker: closed")
        self.closed.emit()


# ==================================================================
# GUI-thread facade: IDSCamera (implements ICamera)
# ==================================================================
class IDSCamera(ICamera):
    """GUI-thread facade. All heavy lifting lives in _StreamWorker (QThread)."""

    # (Signals inherited from ICamera)

    # Mailbox bridges (GUI → worker)
    send_roi = Signal(object)
    send_timing = Signal(object)
    send_gains = Signal(object)
    send_full = Signal()
    send_desat = Signal(float, int)

    def __init__(self) -> None:
        super().__init__()
        self._st = _State()
        self._device = None
        self._remote = None
        self._datastream = None
        self._node_map = None
        self._worker: Optional[_StreamWorker] = None
        self._thread: Optional[QThread] = None
        self._last_roi: dict = {}
        self._zoom_prev_roi: Optional[Tuple[int, int, int, int]] = None  # (W,H,X,Y)

        self.roi.connect(self._remember_roi)

    def _on_started(self) -> None:
        try:
            self._st.acquiring = True
        except Exception:
            pass
        self.started.emit()

    def _on_stopped(self) -> None:
        try:
            self._st.acquiring = False
        except Exception:
            pass
        self.stopped.emit()

    # ---------- facade lifecycle ----------
    @Slot()
    def open(self) -> None:
        log.info("[UI] open()")
        if self._st.open:
            log.debug("[UI] open ignored: already open")
            return
        if _IDS_IMPORT_ERROR is not None:
            self.error.emit(f"ids_peak import failed: {_IDS_IMPORT_ERROR}")
            return
        try:
            ids.Library.Initialize()
            dm = ids.DeviceManager.Instance()
            dm.Update()
            devs = dm.Devices()
            if devs.empty():
                self.error.emit("No device found!")
                self._library_close_safe()
                return
            chosen = next((d for d in devs if d.IsOpenable()), None)
            if chosen is None:
                self.error.emit("Device could not be opened!")
                self._library_close_safe()
                return
            try:
                self._device = chosen.OpenDevice(ids.DeviceAccessType_Exclusive)
            except Exception:
                self._device = chosen.OpenDevice(ids.DeviceAccessType_Control)
            self._remote = self._device.RemoteDevice()
            self._node_map = self._remote.NodeMaps()[0]
            streams = self._device.DataStreams()
            if streams.empty():
                self.error.emit("Device has no DataStream!")
                self._device = None
                self._library_close_safe()
                return
            self._datastream = streams[0].OpenDataStream()

            # worker thread
            self._thread = QThread(self)
            self._worker = _StreamWorker(self._node_map, self._datastream)
            self._worker.moveToThread(self._thread)

            # bridges: worker → facade
            self._worker.started.connect(self._on_started, Qt.QueuedConnection)
            self._worker.stopped.connect(self._on_stopped, Qt.QueuedConnection)
            self._worker.closed.connect(self.closed, Qt.QueuedConnection)

            self._worker.frame.connect(self.frame, Qt.QueuedConnection)
            self._worker.error.connect(self.error, Qt.QueuedConnection)
            self._worker.roi.connect(self.roi, Qt.QueuedConnection)
            self._worker.timing.connect(self.timing, Qt.QueuedConnection)
            self._worker.gains.connect(self.gains, Qt.QueuedConnection)
            self._worker.desaturated.connect(self.desaturated, Qt.QueuedConnection)
            self._worker.auto_desat_started.connect(self.auto_desat_started, Qt.QueuedConnection)
            self._worker.auto_desat_finished.connect(self.auto_desat_finished, Qt.QueuedConnection)

            # mailboxes
            self.send_roi.connect(self._worker.enqueue_roi, Qt.DirectConnection)
            self.send_timing.connect(self._worker.enqueue_timing, Qt.DirectConnection)
            self.send_gains.connect(self._worker.enqueue_gains, Qt.DirectConnection)
            self.send_full.connect(self._worker.enqueue_full_sensor, Qt.DirectConnection)
            self.send_desat.connect(self._worker.enqueue_auto_desat, Qt.DirectConnection)

            self._thread.start()

            self._st.open = True
            name = chosen.DisplayName() if hasattr(chosen, "DisplayName") else "IDS camera"
            log.info("[UI] opened: %s", name)
            self.opened.emit(str(name))

            # Enforce RAW+Mono12 once on open (idle)
            self.force_raw_mono12()

            # initial snapshots
            QMetaObject.invokeMethod(self._worker, "query_roi", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self._worker, "query_gains", Qt.QueuedConnection)
        except Exception as e:
            self.error.emit(f"Failed to open device: {e}")
            log.exception("open() failed")
            self._cleanup_all()

    @Slot()
    def start(self) -> None:
        log.info("[UI] start(); acquiring=%s", self._st.acquiring)
        if not self._st.open or not self._worker:
            log.debug("[UI] start ignored: not open or no worker")
            return
        if self._st.acquiring:
            log.debug("[UI] start ignored: already acquiring")
            return
        QMetaObject.invokeMethod(self._worker, "start", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._worker, "query_gains", Qt.QueuedConnection)

    @Slot()
    def stop(self) -> None:
        log.info("[UI] stop(); acquiring=%s", self._st.acquiring)
        if not self._worker:
            log.debug("[UI] stop ignored: no worker")
            return
        QMetaObject.invokeMethod(self._worker, "set_stop_flag", Qt.DirectConnection)
        QMetaObject.invokeMethod(self._worker, "request_stop", Qt.QueuedConnection)

    @Slot()
    def close(self) -> None:
        if not self._st.open:
            return
        # Stop first if still acquiring
        if self._st.acquiring and self._worker:
            QMetaObject.invokeMethod(self._worker, "set_stop_flag", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self._worker, "request_stop", Qt.QueuedConnection)
        if self._worker:
            QMetaObject.invokeMethod(self._worker, "close", Qt.BlockingQueuedConnection)
        self._cleanup_all()

    # ---------- facade controls ----------
    @Slot()
    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        if not self._worker:
            log.debug("[UI] set_roi ignored: no worker")
            return
        log.info("[UI] set_roi(w=%s, h=%s, x=%s, y=%s)", w, h, x, y)
        payload = {"Width": w, "Height": h, "OffsetX": x, "OffsetY": y}
        self.send_roi.emit(payload)
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_roi", Qt.QueuedConnection)
        else:
            QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)

    @Slot()
    def full_sensor(self) -> None:
        if not self._worker:
            log.debug("[UI] full_sensor ignored: no worker")
            return
        self.send_full.emit()
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_roi", Qt.QueuedConnection)

    @Slot()
    def set_timing(self, fps: Optional[float], exposure_ms: Optional[float]) -> None:
        if not self._worker:
            log.debug("[UI] set_timing ignored: no worker")
            return
        log.info("[UI] set_timing(fps=%s, exposure_ms=%s)", fps, exposure_ms)
        payload: Dict[str, float] = {}
        if fps is not None:
            try:
                payload["fps"] = float(fps)
            except Exception:
                pass
        if exposure_ms is not None:
            try:
                payload["exposure_ms"] = float(exposure_ms)
            except Exception:
                pass
        self.send_timing.emit(payload)
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_timing", Qt.QueuedConnection)

    @Slot()
    def set_gains(self, analog: Optional[float], digital: Optional[float]) -> None:
        if not self._worker:
            log.debug("[UI] set_gains ignored: no worker")
            return
        log.info("[UI] set_gains(analog=%s, digital=%s)", analog, digital)
        payload: Dict[str, Optional[float]] = {}
        if analog is not None:
            payload["analog"] = float(analog)
        if digital is not None:
            payload["digital"] = float(digital)
        self.send_gains.emit(payload)
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_gains", Qt.QueuedConnection)

    @Slot()
    def refresh_timing(self) -> None:
        if not self._worker:
            return
        QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)

    @Slot()
    def refresh_gains(self) -> None:
        if not self._worker:
            log.debug("[UI] refresh_gains ignored: no worker")
            return
        if self._st.acquiring:
            self.send_gains.emit({"__refresh__": True})
        else:
            QMetaObject.invokeMethod(self._worker, "query_gains", Qt.QueuedConnection)

    @Slot(float, int)
    def auto_desaturate(self, target_frac: float = 0.85, max_iters: int = 5) -> None:
        log.info("[UI] auto_desaturate(target=%s, steps=%s)", target_frac, max_iters)
        if not self._worker:
            return
        self.send_desat.emit(float(target_frac), int(max_iters))
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_desat", Qt.QueuedConnection)

    # ---------- zoom helpers ----------
    @Slot(dict)
    def _remember_roi(self, d: dict) -> None:
        try:
            out = {}
            for k in ("Width", "Height", "OffsetX", "OffsetY"):
                v = d.get(k)
                if v is not None:
                    out[k] = int(round(float(v)))
            if out:
                self._last_roi = out
        except Exception:
            pass

    @Slot(object)
    def set_zoom_roi(self, xywh: object) -> None:
        """Request a small ROI around a spot: (x, y, w, h)."""
        if not isinstance(xywh, (tuple, list)) or len(xywh) != 4:
            log.debug("[UI] set_zoom_roi ignored (bad tuple): %s", xywh)
            return
        try:
            x, y, w, h = [int(round(float(v))) for v in xywh]
        except Exception:
            log.debug("[UI] set_zoom_roi ignored (parse failed): %s", xywh)
            return

        if self._zoom_prev_roi is None:
            lr = self._last_roi
            if all(k in lr for k in ("Width", "Height", "OffsetX", "OffsetY")):
                self._zoom_prev_roi = (lr["Width"], lr["Height"], lr["OffsetX"], lr["OffsetY"])
            else:
                if self._worker:
                    QMetaObject.invokeMethod(self._worker, "query_roi", Qt.QueuedConnection)

        self.set_roi(w, h, x, y)

    @Slot()
    def clear_zoom_roi(self) -> None:
        """Restore ROI that was active before the first set_zoom_roi()."""
        if self._zoom_prev_roi is None:
            log.debug("[UI] clear_zoom_roi: nothing to restore")
            return
        w, h, x, y = self._zoom_prev_roi
        self._zoom_prev_roi = None
        self.set_roi(w, h, x, y)

    @Slot()
    def force_raw_mono12(self) -> None:
        """Public facade call; runs in worker thread. Safe only when idle."""
        if not self._worker:
            self.error.emit("Format guard ignored: no worker.")
            return
        if self._st.acquiring:
            self.error.emit("Format change requires camera to be stopped.")
            return
        # Run synchronously so open() can show immediate errors
        QMetaObject.invokeMethod(self._worker, "force_raw_mono12", Qt.BlockingQueuedConnection)

    # ---------- cleanup ----------
    def _cleanup_all(self) -> None:
        try:
            if self._thread:
                self._thread.quit()
                if not self._thread.wait(3000):
                    try:
                        self._thread.terminate()
                        self._thread.wait(1000)
                    except Exception:
                        pass
        except Exception:
            pass
        self._worker = None
        self._thread = None
        self._datastream = None
        self._remote = None
        self._device = None
        self._library_close_safe()
        self._st = _State(open=False, acquiring=False)
        log.info("[UI] cleanup complete")

    def _library_close_safe(self) -> None:
        try:
            if ids is not None:
                ids.Library.Close()
        except Exception:
            pass
