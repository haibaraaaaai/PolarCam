"""
Full-frame-to-disk writer for PolarCam.

Writes raw uint16 frames in chunked .npy files:

    <out_dir>/<prefix>_chunk0000.npy   shape (N, H, W) uint16
    <out_dir>/<prefix>_meta.json

Disk I/O runs in a background thread so cam.frame is never stalled.

Usage::

    writer = FrameWriter()
    writer.start(out_dir="recordings/run1", base_name="run1", chunk_len=50)
    cam.frame.connect(writer.record)
    ...
    writer.stop()
    cam.frame.disconnect(writer.record)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from polarcam.hardware import MOSAIC_LAYOUT_STR


# ---------------------------------------------------------------------------
# Background shard writer
# ---------------------------------------------------------------------------

class _FrameShardWriter(QObject):
    """Receives pre-stacked (N, H, W) uint16 chunks and writes them to disk."""

    finished = Signal()
    error = Signal(str)

    def __init__(self, out_dir: str, prefix: str) -> None:
        super().__init__()
        self._out_dir = out_dir
        self._prefix = prefix
        self._q: Deque[np.ndarray] = deque()
        self._cond = threading.Condition()
        self._closing = False
        self._idx = 0

    @Slot(object)
    def enqueue(self, chunk: np.ndarray) -> None:
        with self._cond:
            self._q.append(chunk)
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
                fn = os.path.join(
                    self._out_dir, f"{self._prefix}_chunk{self._idx:04d}.npy"
                )
                np.save(fn, chunk)
                self._idx += 1
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class FrameWriter(QObject):
    """
    Accumulates raw frames from ``cam.frame`` and writes them to disk.

    Signals
    -------
    started()
        Emitted once acquisition begins.
    stopped()
        Emitted once the writer has flushed and the disk thread has exited.
    progress(int)
        Total frames buffered so far (emitted on every frame).
    error(str)
        Disk I/O error from the background thread.
    """

    started = Signal()
    stopped = Signal()
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._running = False
        self._chunk_len = 20
        self._max_frames = 0
        self._buf: list[np.ndarray] = []
        self._ts_buf: list[float] = []
        self._total = 0
        self._fps_configured: float = 0.0
        self._out_dir: str = ""
        self._prefix: str = ""
        self._writer: Optional[_FrameShardWriter] = None
        self._writer_thr: Optional[QThread] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frames_recorded(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        out_dir: str,
        base_name: str,
        chunk_len: int = 20,
        max_frames: int = 0,
        fps: float = 0.0,
    ) -> None:
        """
        Begin recording.

        Parameters
        ----------
        out_dir : str
            Directory to write files into (created if missing).
        base_name : str
            Prefix used for all output files.
        chunk_len : int
            Frames per .npy file.  Smaller = more files but lower peak memory.
        max_frames : int
            Stop automatically after this many frames.  0 = record until stop().
        fps : float
            Camera FPS at the time recording starts (stored in metadata).
        """
        if self._running:
            return

        self._chunk_len = max(1, int(chunk_len))
        self._max_frames = max(0, int(max_frames))
        self._fps_configured = max(0.0, float(fps))
        self._buf = []
        self._ts_buf = []
        self._total = 0

        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{base_name}_{ts}"
        self._out_dir = out_dir
        self._prefix = prefix

        meta = {
            "base_name": base_name,
            "started_utc": ts,
            "fps_configured": self._fps_configured,
            "chunk_len": self._chunk_len,
            "mosaic_layout": {
                "(row%2, col%2)": MOSAIC_LAYOUT_STR
            },
        }
        try:
            with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except OSError:
            pass

        self._writer = _FrameShardWriter(out_dir, prefix)
        self._writer_thr = QThread()
        self._writer.moveToThread(self._writer_thr)
        self._writer_thr.started.connect(self._writer.run)
        self._writer.finished.connect(self._writer_thr.quit)
        self._writer.error.connect(self.error)
        self._writer_thr.finished.connect(self._on_writer_thread_done)
        self._writer_thr.start()

        self._running = True
        self.started.emit()

    def stop(self) -> None:
        """Flush the current buffer and signal the disk thread to finish.

        Returns immediately; ``stopped`` is emitted once the thread exits.
        """
        if not self._running:
            return
        self._running = False
        self._flush()
        # Save per-frame timestamps (small, fast — fine to do on main thread)
        if self._ts_buf and self._out_dir and self._prefix:
            try:
                ts_arr = np.array(self._ts_buf, dtype=np.float64)
                np.save(os.path.join(self._out_dir, f"{self._prefix}_timestamps.npy"), ts_arr)
            except OSError:
                pass
        self._ts_buf = []
        if self._writer is not None:
            self._writer.close()  # wake the writer loop so it can drain and exit
        # Do NOT block here — _on_writer_thread_done() handles teardown

    @Slot()
    def _on_writer_thread_done(self) -> None:
        """Called in the main thread when the background writer thread exits."""
        self._writer = None
        self._writer_thr = None
        self.stopped.emit()

    # ------------------------------------------------------------------
    # Frame slot — connect directly to cam.frame
    # ------------------------------------------------------------------

    @Slot(object)
    def record(self, frame_obj: object) -> None:
        """Accept one frame from cam.frame and buffer it."""
        if not self._running:
            return
        arr = np.asarray(frame_obj)
        if arr.ndim != 2:
            return
        self._ts_buf.append(time.perf_counter())
        self._buf.append(arr.astype(np.uint16, copy=True))
        self._total += 1
        self.progress.emit(self._total)

        if len(self._buf) >= self._chunk_len:
            self._flush()

        if self._max_frames > 0 and self._total >= self._max_frames:
            self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        if not self._buf or self._writer is None:
            return
        chunk = np.stack(self._buf, axis=0)  # (N, H, W) uint16
        self._buf = []
        self._writer.enqueue(chunk)
