from __future__ import annotations
"""
Camera backend interface (Qt/ABC).

Implementations should emit:
- frame(object np.ndarray[H,W] dtype=uint16)
- opened(str), started(), stopped(), closed()
- error(str)
- roi(dict), timing(dict)
- (optional) gains(dict), desaturated(dict), auto_desat_started(), auto_desat_finished()

All timing uses *milliseconds* for exposure on the facade methods.
"""

from typing import Optional, Tuple
from abc import ABCMeta, abstractmethod
from PySide6.QtCore import QObject, Signal

# Combine PySide's metaclass with ABCMeta to avoid the conflict.
class _QABCMeta(type(QObject), ABCMeta):
    pass


class ICamera(QObject, metaclass=_QABCMeta):
    # Core streaming/data
    frame  = Signal(object)

    # Lifecycle
    opened = Signal(str)
    started = Signal()
    stopped = Signal()
    closed = Signal()
    error = Signal(str)

    # Telemetry
    roi = Signal(dict)
    timing = Signal(dict)

    # Optional (if backend supports it)
    gains = Signal(dict)
    desaturated = Signal(dict)
    auto_desat_started = Signal()
    auto_desat_finished = Signal()

    # ---- required API ----
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...

    # Controls
    @abstractmethod
    def set_roi(self, w: int, h: int, x: int, y: int) -> None: ...
    @abstractmethod
    def full_sensor(self) -> None: ...
    @abstractmethod
    def set_timing(self, fps: Optional[float] = None, exposure_ms: Optional[float] = None) -> None: ...
    @abstractmethod
    def set_zoom_roi(self, roi_xywh: Optional[Tuple[int, int, int, int]]) -> None: ...

    # Nice-to-have API used by the app (explicit here for clarity)
    @abstractmethod
    def set_gains(self, analog: Optional[float], digital: Optional[float]) -> None: ...
    @abstractmethod
    def refresh_gains(self) -> None: ...
    @abstractmethod
    def refresh_timing(self) -> None: ...
    @abstractmethod
    def auto_desaturate(self, target_frac: float = 0.85, max_iters: int = 5) -> None: ...
